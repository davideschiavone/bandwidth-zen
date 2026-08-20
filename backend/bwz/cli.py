"""bwz command-line interface (typer)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console
from rich.table import Table

import bwz
from bwz.analysis import (
    analyze,
    idealised,
    machine_model,
    rank_operations,
    suggestions,
)
from bwz.analysis.compare import head_to_head, prefill_crossover
from bwz.analysis.dataflow import plan_dataflow
from bwz.analysis.pipeline import Lane, PipelineTrace, build_trace
from bwz.analysis.roofline import compute_dtype
from bwz.graph import build_graph
from bwz.graph.ops import GraphPhase
from bwz.kernels import encoder_layer_kernel, matmul_kernel
from bwz.operators.base import cost_of
from bwz.report import Bound, Report
from bwz.spec import (
    CNNSpec,
    CustomSpec,
    DType,
    FFNType,
    HardwareSpec,
    MatmulSpec,
    NormType,
    SpecLoadError,
    TransformerSpec,
    bytes_per_element,
    iter_chips,
    iter_models,
    load_chip,
    load_model,
)
from bwz.spec.deployment import AStrategy, AttentionImpl, BDataflow, DeploymentSpec, Phase
from bwz.spec.loaders import AnyModelSpec
from bwz.units import format_bandwidth, format_bytes, format_quantity, format_time

app = typer.Typer(
    name="bwz",
    help="bandwidth-zen: analytical performance model for neural-network inference.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    epilog=(
        "[bold]Typical order:[/bold] [cyan]matmul[/cyan] probes one GEMM on one chip — the "
        "smallest roofline check, and where the dataflow strategy flags "
        "([cyan]--a-strategy[/cyan]/[cyan]--b-dataflow[/cyan], see [cyan]docs/CLI.md[/cyan] §2.5) "
        "live. [cyan]run[/cyan] costs a full model. [cyan]compare[/cyan] puts chips head to head. "
        "[cyan]list[/cyan] shows the bundled chip/model ids these all take.\n\n"
        "Each command prints numbers, derivations and an assumptions drawer — nothing here plots. "
        "For the same run as a zoomable picture: "
        "[cyan]uv run python scripts/plot_pipeline.py --help[/cyan]."
    ),
)
console = Console()

# Named --help panels (Rich groups options under a heading instead of one flat
# list), shared across commands so the same knob always lands under the same
# name. Order here has no effect; a panel's position in --help follows where
# its first option is declared in the command's signature.
PANEL_WORKLOAD = "workload"
PANEL_SHAPE = "shape"
PANEL_ARCHITECTURE = "architecture"
PANEL_PRECISION = "precision"
PANEL_DEPLOYMENT = "deployment"
PANEL_DATAFLOW = "dataflow strategy (docs/CLI.md §2.5)"
PANEL_DATAFLOW_INERT = "dataflow strategy — inert on a network (docs/CLI.md §2.5)"
PANEL_OUTPUT = "output"


class Catalog(StrEnum):
    CHIPS = "chips"
    MODELS = "models"
    ALL = "all"


@app.callback()
def main() -> None:
    """bandwidth-zen CLI. Run a subcommand; see --help."""


@app.command()
def version() -> None:
    """Print the bwz version."""
    typer.echo(f"bwz {bwz.__version__}")


def _headline_dtype(chip: HardwareSpec) -> DType:
    """The dtype a chip's peak figure is most naturally quoted in."""
    for candidate in (DType.FP16, DType.INT8, DType.BF16, DType.FP32):
        if chip.supports(candidate):
            return candidate
    return chip.compute_units[0].supported_dtypes[0]


def _provenance(spec: HardwareSpec | AnyModelSpec) -> str:
    if spec.hypothetical:
        return "[yellow]hypothetical[/yellow]"
    if spec.estimates:
        return f"[cyan]{len(spec.estimates)} est.[/cyan]"
    return "[green]sourced[/green]"


def _chip_table() -> Table:
    table = Table(title="Chips", title_justify="left", header_style="bold")
    table.add_column("id")
    table.add_column("name")
    table.add_column("peak", justify="right")
    table.add_column("dtype")
    table.add_column("DRAM BW", justify="right")
    table.add_column("DRAM", justify="right")
    table.add_column("on-chip", justify="right")
    table.add_column("ridge", justify="right")
    table.add_column("provenance")
    for chip in iter_chips():
        dtype = _headline_dtype(chip)
        peak = chip.peak_flops_per_s(dtype)
        table.add_row(
            chip.id,
            chip.name,
            format_quantity(peak, "OP/s" if dtype in (DType.INT8, DType.INT4) else "FLOP/s"),
            dtype.value,
            format_bandwidth(chip.dram.bandwidth_bytes_per_s),
            format_bytes(chip.dram.capacity_bytes),
            format_bytes(chip.on_chip_capacity_bytes),
            f"{chip.ridge_point_flops_per_byte(dtype):.0f}",
            _provenance(chip),
        )
    return table


def _model_row(model: AnyModelSpec) -> tuple[str, ...]:
    if isinstance(model, TransformerSpec):
        p = model.effective_params
        # h=<query>/<kv>x<head_dim>. Showing kv_heads alone hid both the GQA
        # ratio and the fact that heads x head_dim need not equal hidden --
        # Gemma-3-4B is 8x256 = 2048 against a hidden of 2560.
        shape = (
            f"L={p.layers} d={p.hidden} ffn={p.ffn_hidden} "
            f"h={p.heads}/{p.effective_kv_heads}x{p.effective_head_dim}"
        )
        params = format_quantity(model.headline_parameter_count(), "", precision=4).strip()
        return (model.id, model.name, model.family.value, params, shape, _provenance(model))
    if isinstance(model, CNNSpec):
        shape = f"{model.input.channels}x{model.input.height}x{model.input.width}"
        return (
            model.id,
            model.name,
            model.family.value,
            f"{len(model.layers)} layers",
            shape,
            _provenance(model),
        )
    if isinstance(model, MatmulSpec):
        params = format_quantity(float(model.parameter_count()), "", precision=4).strip()
        return (
            model.id,
            model.name,
            model.family.value,
            params,
            f"M={model.m} N={model.n} K={model.k}",
            _provenance(model),
        )
    assert isinstance(model, CustomSpec)
    return (
        model.id,
        model.name,
        model.family.value,
        f"{len(model.ops)} ops",
        "",
        _provenance(model),
    )


def _model_table() -> Table:
    table = Table(title="Models", title_justify="left", header_style="bold")
    table.add_column("id")
    table.add_column("name")
    table.add_column("family")
    table.add_column("params", justify="right")
    table.add_column("shape")
    table.add_column("provenance")
    for model in iter_models():
        table.add_row(*_model_row(model))
    return table


def _report_for(
    model: AnyModelSpec, chip_id: str, deployment: DeploymentSpec, *, ideal: bool = False
) -> Report:
    """Analyse an already-constructed spec. The wall-clock stamp is applied here,
    outside the pure core (CLAUDE.md #3)."""
    report = analyze(model, _chip_for(chip_id, ideal), deployment)
    return replace(report, meta=replace(report.meta, generated_at=datetime.now(UTC).isoformat()))


def _run_report(
    model_id: str, chip_id: str, deployment: DeploymentSpec, *, ideal: bool = False
) -> Report:
    return _report_for(load_model(model_id), chip_id, deployment, ideal=ideal)


def _summary_table(report: Report) -> Table:
    table = Table(title=f"{report.meta.model_name} on {report.meta.chip_name}", box=box.SIMPLE)
    table.add_column("phase")
    table.add_column("latency", justify="right")
    table.add_column("bound")
    table.add_column("util", justify="right")
    table.add_column("t_dram", justify="right")
    table.add_column("t_compute", justify="right")
    table.add_column("t_fixed", justify="right")
    for phase in report.phases:
        table.add_row(
            phase.phase.value,
            format_time(phase.latency_s),
            _colour_bound(phase.bound),
            f"{phase.utilization:.2%}",
            format_time(phase.t_dram_s),
            format_time(phase.t_compute_s),
            format_time(phase.t_fixed_s),
        )
    return table


def _colour_bound(bound: Bound) -> str:
    colour = {
        Bound.DRAM_BW_BOUND: "yellow",
        Bound.COMPUTE_BOUND: "green",
        Bound.LATENCY_BOUND: "red",
    }[bound]
    return f"[{colour}]{bound.value}[/{colour}]"


def _memory_table(report: Report) -> Table:
    plan = report.memory
    table = Table(title="Memory", box=box.SIMPLE)
    table.add_column("item")
    table.add_column("bytes", justify="right")
    table.add_row("weights", format_bytes(plan.weight_bytes))
    table.add_row("KV cache", format_bytes(plan.kv_cache_bytes))
    table.add_row("peak activations", format_bytes(plan.peak_activation_bytes))
    table.add_row("[bold]total[/bold]", f"[bold]{format_bytes(plan.total_bytes)}[/bold]")
    table.add_row("usable DRAM", format_bytes(plan.usable_dram_bytes))
    table.add_row("on-chip", format_bytes(plan.on_chip_capacity_bytes))
    table.add_row("weight residency", f"{plan.resident_fraction:.2%}")
    table.add_row("activation residency", f"{plan.activation_resident_fraction:.0%}")
    table.add_row("double buffered", "yes" if plan.double_buffered else "no")
    return table


def _dataflow_options(
    a_strategy: AStrategy,
    b_dataflow: BDataflow,
    a_residency_tiles: int | None,
    a_prefetch_depth: int | None,
    iterations: int,
) -> dict[str, object]:
    """The single-matmul dataflow fields, as a fragment to merge into a
    ``DeploymentSpec`` dict. Shared by ``matmul`` and ``run`` so the two
    commands cannot drift onto different field names.
    """
    return {
        "a_strategy": a_strategy,
        "b_dataflow": b_dataflow,
        "a_residency_tiles": a_residency_tiles,
        "a_prefetch_depth": a_prefetch_depth,
        "iterations": iterations,
    }


@app.command()
def run(
    model: str = typer.Option(
        ..., "--model", "-m", help="Model profile id or path", rich_help_panel=PANEL_WORKLOAD
    ),
    chip: str = typer.Option(
        ..., "--chip", "-c", help="Chip profile id or path", rich_help_panel=PANEL_WORKLOAD
    ),
    batch: int = typer.Option(1, "--batch", "-b", rich_help_panel=PANEL_DEPLOYMENT),
    input_tokens: int = typer.Option(2048, "--input-tokens", rich_help_panel=PANEL_DEPLOYMENT),
    output_tokens: int = typer.Option(256, "--output-tokens", rich_help_panel=PANEL_DEPLOYMENT),
    context: int | None = typer.Option(
        None,
        "--context",
        help="KV context; defaults to in+out",
        rich_help_panel=PANEL_DEPLOYMENT,
    ),
    weights: DType = typer.Option(
        DType.FP16, "--weights", help="Weight precision", rich_help_panel=PANEL_DEPLOYMENT
    ),
    phase: Phase = typer.Option(Phase.BOTH, "--phase", rich_help_panel=PANEL_DEPLOYMENT),
    attention: AttentionImpl = typer.Option(
        AttentionImpl.FLASH2, "--attention", rich_help_panel=PANEL_DEPLOYMENT
    ),
    a_strategy: AStrategy = typer.Option(
        AStrategy.STAGE,
        "--a-strategy",
        help="How A is loaded for a lone matmul: stage (once per k-slice, D33), "
        "stream (per tile, D31) or whole (all of A before the first tile). Inert on a "
        "network, whose activations are governed by inter-operation residency instead.",
        rich_help_panel=PANEL_DATAFLOW_INERT,
    ),
    b_dataflow: BDataflow = typer.Option(
        BDataflow.WRITE_AHEAD,
        "--b-dataflow",
        help="When B's array write lands: write-ahead (a wave early, hidden behind "
        "compute), on-demand (at compute, exposed) or persistent (once, never "
        "displaced — needs tiles <= units * weight_sets).",
        rich_help_panel=PANEL_DATAFLOW_INERT,
    ),
    a_residency_tiles: int | None = typer.Option(
        None,
        "--a-residency-tiles",
        help="Override tiles served per A staging event under stage/whole; must be a "
        "power-of-2 divisor of NTILES_PER_KS (clamped otherwise). Default: the whole "
        "k-slice.",
        rich_help_panel=PANEL_DATAFLOW_INERT,
    ),
    a_prefetch_depth: int | None = typer.Option(
        None,
        "--a-prefetch-depth",
        help="Override the double-buffered staging depth for A. Schedule-only — "
        "changes no byte count. Default: derived from double buffering, as today.",
        rich_help_panel=PANEL_DATAFLOW_INERT,
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        help="Invocations this report represents. Only b_dataflow=persistent reads "
        "it, amortising B's write over a resident weight set a repeat invocation "
        "would not have to rewrite.",
        rich_help_panel=PANEL_DATAFLOW_INERT,
    ),
    show_ops: int = typer.Option(
        0,
        "--show-ops",
        help="Show the N most expensive operations",
        rich_help_panel=PANEL_OUTPUT,
    ),
    ideal: bool = typer.Option(
        False,
        "--ideal",
        help="Zero every unfitted calibration constant — both efficiencies and the "
        "per-dispatch overhead: a hardware ceiling, not a prediction",
        rich_help_panel=PANEL_OUTPUT,
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw Report as JSON", rich_help_panel=PANEL_OUTPUT
    ),
) -> None:
    """Predict how a model runs on a chip."""
    try:
        deployment = DeploymentSpec.model_validate(
            {
                "batch": batch,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "kv_context_tokens": context,
                "phase": phase,
                "attention_impl": attention,
                "precision": {"weights": weights, "activations": weights, "kv_cache": weights},
                **_dataflow_options(
                    a_strategy, b_dataflow, a_residency_tiles, a_prefetch_depth, iterations
                ),
            }
        )
        report = _run_report(model, chip, deployment, ideal=ideal)
    except SpecLoadError as exc:
        console.print(f"[red]bwz:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(report.to_json())
        return

    if not report.feasible:
        console.print("[red]Infeasible.[/red]")
        for reason in report.infeasibility:
            console.print(f"  • {reason}")
        raise typer.Exit(code=2)

    summary = report.summary
    assert summary is not None
    _present(report, chip, weights, ideal=ideal, show_ops=show_ops)


def _present(
    report: Report,
    chip_id: str,
    weights: DType,
    *,
    ideal: bool,
    show_ops: int,
) -> None:
    """The body of a report: summary, memory, why, operations, assumptions.

    Shared so that every command that produces a ``Report`` presents it the same
    way — a reader should not have to learn a second layout because the shape
    came from arguments rather than a profile.
    """
    summary = report.summary
    assert summary is not None
    console.print(_summary_table(report))
    if summary.ttft_s is not None:
        console.print(f"  TTFT      {format_time(summary.ttft_s)}")
    if summary.tpot_s is not None:
        console.print(
            f"  TPOT      {format_time(summary.tpot_s)}   "
            f"[bold]{summary.tokens_per_s:.1f} tok/s[/bold]"
        )
    console.print(f"  total     {format_time(summary.latency_s)}")
    console.print(
        f"  achieved  {format_quantity(summary.achieved_flops_per_s, 'OP/s')} "
        f"of {format_quantity(summary.peak_flops_per_s, 'OP/s')} "
        f"({summary.utilization:.2%})"
    )
    console.print()
    console.print(_memory_table(report))

    console.print(f"\n[bold]Why[/bold]  (confidence: {report.confidence.value})")
    for margin in report.flip_margins:
        flag = " [yellow](rests on an estimated input)[/yellow]" if margin.rests_on_estimate else ""
        console.print(f"  • {margin.description}{flag}")
    machine = machine_model(
        _chip_for(chip_id, ideal), compute_dtype(_chip_for(chip_id, ideal), weights, weights)
    )
    for phase_result in report.phases:
        for hint in suggestions(phase_result, machine):
            console.print(f"  → {hint}")

    if show_ops:
        console.print("\n[bold]Most expensive operations[/bold]")
        for phase_result in report.phases:
            for line in rank_operations(phase_result, limit=show_ops):
                console.print(f"  {phase_result.phase.value}: {line}")

    console.print(f"\n[bold]Assumptions[/bold] ({len(report.assumptions)})")
    for assumption in report.assumptions:
        console.print(f"  • {assumption}", highlight=False)


def _chip_for(chip_id: str, ideal: bool) -> HardwareSpec:
    chip = load_chip(chip_id)
    return idealised(chip) if ideal else chip


def _pipeline_note(trace: PipelineTrace) -> str:
    """One line on what the schedule costs that the roofline number does not."""
    steps = (
        f"{trace.steps} steps drawn, coalesced from {trace.tiles} tiles"
        if trace.coalesced
        else f"{trace.steps} tile steps"
    )
    if not trace.double_buffered:
        return (
            f"{steps}; no double buffer, so each load waits for the previous tile's "
            f"arithmetic and the span is load + compute exactly."
        )
    share = trace.fill_drain_s / trace.total_s if trace.total_s > 0 else 0.0
    return (
        f"{steps}, double buffered. Span {format_time(trace.total_s)} against a reported "
        f"{format_time(trace.reported_latency_s)}: the extra {format_time(trace.fill_drain_s)} "
        f"({share:.1%}) is pipeline fill/drain, which max(load, compute) omits."
    )


def _lane_table(trace: PipelineTrace) -> Table:
    """Who was busy, and for how long.

    DRAM and the array are single serial resources, so their occupancy is a duty
    cycle and cannot exceed 100%. SRAM is not a serial resource — it is *n*
    buffers — so the same arithmetic gives a depth, and printing it as a
    percentage made "two tiles resident throughout" read as "198% busy". The
    column therefore carries its own unit per row.
    """
    table = Table(title="Pipeline", box=box.SIMPLE)
    table.add_column("lane")
    table.add_column("busy", justify="right")
    table.add_column("occupancy", justify="right")
    table.add_column("what it was doing")
    busy = trace.busy_s
    concurrency = trace.concurrency
    descriptions = {
        Lane.DRAM: "operand tiles in, results out, across the one modelled link",
        Lane.SRAM: "tile buffers held from fetch to use",
        Lane.CORE: "matrix arithmetic, plus the dispatches",
        Lane.VECTOR: "norms, activations, residuals — not the matrix engine",
    }
    for lane in Lane:
        mean, peak = concurrency[lane]
        if lane is Lane.SRAM:
            occupancy = f"{mean:.2f} of {peak} buf" if peak else "—"
        else:
            occupancy = f"{mean:.0%} of span"
        table.add_row(lane.value, format_time(busy[lane]), occupancy, descriptions.get(lane, ""))
    return table


@app.command()
def matmul(
    m: int = typer.Option(
        ..., "--m", "-M", help="Rows of operand A; folds batch in", rich_help_panel=PANEL_SHAPE
    ),
    n: int = typer.Option(
        ..., "--n", "-N", help="Columns of operand B", rich_help_panel=PANEL_SHAPE
    ),
    k: int = typer.Option(
        ..., "--k", "-K", help="Contracted (inner) dimension", rich_help_panel=PANEL_SHAPE
    ),
    chip: str = typer.Option(
        ..., "--chip", "-c", help="Chip profile id or path", rich_help_panel=PANEL_SHAPE
    ),
    dtype: DType = typer.Option(
        DType.FP16,
        "--dtype",
        "-d",
        help="Width of both operands, and of the result unless --out",
        rich_help_panel=PANEL_PRECISION,
    ),
    a_dtype: DType | None = typer.Option(
        None, "--a", help="Width of the M x K operand A", rich_help_panel=PANEL_PRECISION
    ),
    b_dtype: DType | None = typer.Option(
        None, "--b", help="Width of the K x N operand B", rich_help_panel=PANEL_PRECISION
    ),
    out_dtype: DType | None = typer.Option(
        None,
        "--out",
        help="Width of the M x N result. Defaults to the wider operand; set int32 or fp32 for a "
        "widening accumulator",
        rich_help_panel=PANEL_PRECISION,
    ),
    a_strategy: AStrategy = typer.Option(
        AStrategy.STAGE,
        "--a-strategy",
        help="How A is loaded: stage (once per k-slice, D33), stream (per tile, D31) "
        "or whole (all of A before the first tile).",
        rich_help_panel=PANEL_DATAFLOW,
    ),
    b_dataflow: BDataflow = typer.Option(
        BDataflow.WRITE_AHEAD,
        "--b-dataflow",
        help="When B's array write lands: write-ahead (a wave early, hidden behind "
        "compute), on-demand (at compute, exposed) or persistent (once, never "
        "displaced — needs tiles <= units * weight_sets).",
        rich_help_panel=PANEL_DATAFLOW,
    ),
    a_residency_tiles: int | None = typer.Option(
        None,
        "--a-residency-tiles",
        help="Override tiles served per A staging event under stage/whole; must be a "
        "power-of-2 divisor of NTILES_PER_KS (clamped otherwise). Default: the whole "
        "k-slice.",
        rich_help_panel=PANEL_DATAFLOW,
    ),
    a_prefetch_depth: int | None = typer.Option(
        None,
        "--a-prefetch-depth",
        help="Override the double-buffered staging depth for A. Schedule-only — "
        "changes no byte count. Default: derived from double buffering, as today.",
        rich_help_panel=PANEL_DATAFLOW,
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        help="Invocations this report represents. Only b_dataflow=persistent reads "
        "it, amortising B's write over a resident weight set a repeat invocation "
        "would not have to rewrite.",
        rich_help_panel=PANEL_DATAFLOW,
    ),
    ideal: bool = typer.Option(
        False,
        "--ideal",
        help="Zero every unfitted calibration constant — both efficiencies and the "
        "per-dispatch overhead: a hardware ceiling, not a prediction",
        rich_help_panel=PANEL_OUTPUT,
    ),
    pipeline: bool = typer.Option(
        True,
        "--pipeline/--no-pipeline",
        help="Show which resource is busy for how long",
        rich_help_panel=PANEL_OUTPUT,
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw Report as JSON", rich_help_panel=PANEL_OUTPUT
    ),
) -> None:
    """Run one A[M,K] x B[K,N] -> C[M,N]: the smallest probe of a chip's roofline.

    Matmul vocabulary throughout — operands A and B and a result C, not weights
    and activations, which mean nothing outside a network. No batch, context or
    phase knobs either: M is the full row count, so a batch of 128 is M=128.

    The result width is the accumulator width and changes bytes only, never
    operations: int8 x int8 -> int32 does exactly the same 2*M*N*K integer
    operations as int8 x int8 -> int8, and writes four times the bytes.
    """
    a = a_dtype if a_dtype is not None else dtype
    b = b_dtype if b_dtype is not None else dtype
    try:
        spec = matmul_kernel(m, n, k, a_dtype=a, b_dtype=b, out_dtype=out_dtype)
        # DeploymentSpec is required by analyze() but a bare matmul reads nothing
        # from it beyond the dataflow strategy flags: the builder takes its widths
        # from the spec (D18) and there is no batch, context or phase to describe.
        deployment = DeploymentSpec.model_validate(
            {
                "batch": 1,
                "input_tokens": 1,
                "output_tokens": 0,
                "phase": Phase.PREFILL,
                **_dataflow_options(
                    a_strategy, b_dataflow, a_residency_tiles, a_prefetch_depth, iterations
                ),
            }
        )
        report = _report_for(spec, chip, deployment, ideal=ideal)
    except SpecLoadError as exc:
        console.print(f"[red]bwz:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(report.to_json())
        return

    if not report.feasible:
        console.print("[red]Infeasible.[/red]")
        for reason in report.infeasibility:
            console.print(f"  • {reason}")
        raise typer.Exit(code=2)

    machine = machine_model(_chip_for(chip, ideal), spec.operand_dtype)
    summary = report.summary
    assert summary is not None
    op = report.phases[0].ops[0]
    result = spec.result_dtype

    table = Table(
        title=(
            # No square brackets: rich would read them as markup tags.
            f"A {m}x{k} {a.value}  x  B {k}x{n} {b.value}  ->  C {m}x{n} {result.value}"
            f"\non {report.meta.chip_name}"
        ),
        box=box.SIMPLE,
    )
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_column("derivation")
    table.add_row(
        "operations",
        format_quantity(op.flops, "OP"),
        f"2 x {m} x {n} x {k} — unchanged by the result width",
    )
    table.add_row(
        "arithmetic runs at",
        spec.operand_dtype.value,
        f"{machine.unit.name} peak {format_quantity(machine.peak_flops_per_s, 'OP/s')}"
        + ("" if a is b else " — the wider operand; both share one datapath"),
    )
    table.add_row(
        "operand A",
        format_bytes(m * k * bytes_per_element(a)),
        f"{m} x {k} x {bytes_per_element(a):g} B",
    )
    table.add_row(
        "operand B",
        format_bytes(k * n * bytes_per_element(b)),
        f"{k} x {n} x {bytes_per_element(b):g} B",
    )
    table.add_row(
        "result C",
        format_bytes(m * n * bytes_per_element(result)),
        f"{m} x {n} x {bytes_per_element(result):g} B"
        + ("" if out_dtype is None else " — widening accumulator"),
    )
    table.add_row(
        "DRAM reads",
        format_bytes(op.dram_read_bytes),
        "A and B, less whatever stays on chip",
    )
    table.add_row(
        "DRAM writes",
        format_bytes(op.dram_write_bytes),
        "C in full — nothing on chip consumes the result, so it must be written",
    )
    table.add_row("DRAM traffic", format_bytes(op.dram_bytes), "reads + writes")
    table.add_row(
        "intensity", f"{op.arithmetic_intensity:.1f} OP/byte", "operations / compulsory traffic"
    )
    table.add_row(
        "ridge point",
        f"{machine.ridge_point:.1f} OP/byte",
        "effective OP/s / effective bytes/s — above it the chip is compute-bound",
    )
    dims = machine.unit.systolic_dims
    table.add_row(
        "shape utilisation",
        f"{op.utilization:.2%}",
        (
            f"systolic tail on a {dims[0]}x{dims[1]} array — geometry, not a derating"
            if dims is not None
            else "profile declares no array geometry, so no tail effect is claimed"
        ),
    )
    table.add_row("t_dram", format_time(op.t_dram_s), "traffic / effective bandwidth")
    table.add_row("t_compute", format_time(op.t_compute_s), "operations / (effective peak x util)")
    table.add_row("t_fixed", format_time(op.t_fixed_s), "one kernel dispatch")
    table.add_row("[bold]latency[/bold]", f"[bold]{format_time(op.latency_s)}[/bold]", "")
    table.add_row("[bold]verdict[/bold]", _colour_bound(op.bound), "")
    console.print(table)

    console.print(
        f"  achieved  {format_quantity(summary.achieved_flops_per_s, 'OP/s')} "
        f"of {format_quantity(summary.peak_flops_per_s, 'OP/s')} "
        f"({summary.utilization:.2%})"
    )

    if pipeline:
        graph = build_graph(spec, deployment, GraphPhase.STATIC)
        dataflow = plan_dataflow(
            graph.ops[0],
            machine,
            _chip_for(chip, ideal),
            deployment,
            a_bytes=cost_of(graph.ops[0], graph.tensors).input_bytes,
        )
        trace = build_trace(
            graph,
            report.phases[0],
            machine,
            double_buffered=report.memory.double_buffered,
            dataflow=dataflow,
        )
        console.print()
        console.print(_lane_table(trace))
        console.print(f"  {_pipeline_note(trace)}", highlight=False)

    console.print(f"\n[bold]Why[/bold]  (confidence: {report.confidence.value})")
    for margin in report.flip_margins:
        flag = " [yellow](rests on an estimated input)[/yellow]" if margin.rests_on_estimate else ""
        console.print(f"  • {margin.description}{flag}")
    for hint in suggestions(report.phases[0], machine):
        console.print(f"  → {hint}")

    console.print(f"\n[bold]Assumptions[/bold] ({len(report.assumptions)})")
    for assumption in report.assumptions:
        console.print(f"  • {assumption}", highlight=False)


@app.command(name="encoder-layer")
def encoder_layer(
    chip: str = typer.Option(
        ..., "--chip", "-c", help="Chip profile id or path", rich_help_panel=PANEL_SHAPE
    ),
    hidden: int = typer.Option(
        8, "--hidden", "-d", help="Model width", rich_help_panel=PANEL_SHAPE
    ),
    heads: int = typer.Option(2, "--heads", help="Attention heads", rich_help_panel=PANEL_SHAPE),
    head_dim: int | None = typer.Option(
        None, "--head-dim", help="Defaults to hidden // heads", rich_help_panel=PANEL_SHAPE
    ),
    ffn: int = typer.Option(16, "--ffn", help="FFN inner width", rich_help_panel=PANEL_SHAPE),
    vocab: int = typer.Option(16, "--vocab", help="Vocabulary size", rich_help_panel=PANEL_SHAPE),
    tokens: int = typer.Option(
        4, "--tokens", "-S", help="Sequence length", rich_help_panel=PANEL_SHAPE
    ),
    batch: int = typer.Option(1, "--batch", "-b", rich_help_panel=PANEL_SHAPE),
    ffn_type: FFNType = typer.Option(
        FFNType.RELU, "--ffn-type", rich_help_panel=PANEL_ARCHITECTURE
    ),
    norm: NormType = typer.Option(NormType.RMSNORM, "--norm", rich_help_panel=PANEL_ARCHITECTURE),
    tie: bool = typer.Option(
        True,
        "--tie/--untie",
        help="Tie the embedding and output tables",
        rich_help_panel=PANEL_ARCHITECTURE,
    ),
    weights: DType = typer.Option(
        DType.FP16, "--weights", help="Precision", rich_help_panel=PANEL_PRECISION
    ),
    ideal: bool = typer.Option(
        False,
        "--ideal",
        help="Zero every unfitted calibration constant — both efficiencies and the "
        "per-dispatch overhead: a hardware ceiling, not a prediction",
        rich_help_panel=PANEL_OUTPUT,
    ),
    show_ops: int = typer.Option(
        20,
        "--show-ops",
        help="Show the N most expensive operations",
        rich_help_panel=PANEL_OUTPUT,
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the raw Report as JSON", rich_help_panel=PANEL_OUTPUT
    ),
) -> None:
    """One encoder layer, sized from the command line, small enough to count by hand.

    The transformer counterpart of `bwz matmul`: the shape is arguments rather
    than a profile, so a dimension can be changed and its effect read straight
    off. One layer, always — that is the point. For anything deeper, write a
    profile and use `bwz run`.

    Bidirectional attention over all S tokens, no KV cache and no LM head: an
    encoder has no later step to reuse a cache for, and what sits on top of it is
    task-specific (docs/CORRECTIONS.md D24).
    """
    try:
        spec = encoder_layer_kernel(
            hidden=hidden,
            heads=heads,
            head_dim=head_dim,
            ffn=ffn,
            vocab=vocab,
            tokens=tokens,
            ffn_type=ffn_type,
            norm=norm,
            tie_embeddings=tie,
        )
        deployment = DeploymentSpec.model_validate(
            {
                "batch": batch,
                "input_tokens": tokens,
                "output_tokens": 0,
                "phase": Phase.PREFILL,
                "precision": {"weights": weights, "activations": weights, "kv_cache": weights},
            }
        )
        report = _report_for(spec, chip, deployment, ideal=ideal)
    except (SpecLoadError, ValidationError) as exc:
        console.print(f"[red]bwz:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(report.to_json())
        return
    if not report.feasible:
        console.print("[red]Infeasible.[/red]")
        for reason in report.infeasibility:
            console.print(f"  • {reason}")
        raise typer.Exit(code=2)

    console.print(_shape_table(spec, deployment))
    _present(report, chip, weights, ideal=ideal, show_ops=show_ops)


def _shape_table(spec: TransformerSpec, deployment: DeploymentSpec) -> Table:
    """Where every parameter is, as a sum a reader can check.

    The point of sizing a model from the command line is watching one term move,
    so the terms are listed rather than only their total.
    """
    p = spec.params
    table = Table(title=f"{spec.name}, S={deployment.input_tokens}", box=box.SIMPLE)
    table.add_column("what")
    table.add_column("parameters", justify="right")
    table.add_column("derivation")
    table.add_row(
        "Q, K, V, O",
        f"{p.attention_params_per_layer():,}",
        f"{p.hidden}x{p.q_width} + 2 x {p.hidden}x{p.kv_width} + {p.q_width}x{p.hidden}",
    )
    table.add_row(
        "FFN",
        f"{p.ffn_params_per_layer():,}",
        f"{p.ffn_type.n_matrices} x {p.hidden} x {p.ffn_hidden}",
    )
    table.add_row(
        "norms",
        f"{p.norm_params_per_layer():,}",
        f"2 x {p.norm.value} over {p.hidden} channels",
    )
    table.add_row("[bold]per layer[/bold]", f"[bold]{p.params_per_layer():,}[/bold]", "")
    table.add_row(
        "embeddings",
        f"{p.embedding_params():,}",
        f"{p.vocab} x {p.hidden}" + (", tied" if p.tie_embeddings else ", untied so x2"),
    )
    table.add_row("final norm", f"{p.norm_params_per_layer() // 2:,}", "")
    table.add_row("[bold]total[/bold]", f"[bold]{spec.parameter_count():,}[/bold]", "")
    return table


@app.command()
def compare(
    chips: str = typer.Option(
        ...,
        "--chips",
        help="Comma-separated chip ids, e.g. chip_a,chip_b",
        rich_help_panel=PANEL_WORKLOAD,
    ),
    models: str = typer.Option(
        ..., "--models", help="Comma-separated model ids", rich_help_panel=PANEL_WORKLOAD
    ),
    batch: int = typer.Option(1, "--batch", "-b", rich_help_panel=PANEL_DEPLOYMENT),
    input_tokens: int = typer.Option(512, "--input-tokens", rich_help_panel=PANEL_DEPLOYMENT),
    output_tokens: int = typer.Option(1, "--output-tokens", rich_help_panel=PANEL_DEPLOYMENT),
    context: int | None = typer.Option(None, "--context", rich_help_panel=PANEL_DEPLOYMENT),
    weights: DType = typer.Option(DType.INT8, "--weights", rich_help_panel=PANEL_DEPLOYMENT),
    crossover: bool = typer.Option(
        True, "--crossover/--no-crossover", rich_help_panel=PANEL_OUTPUT
    ),
    ideal: bool = typer.Option(
        False,
        "--ideal",
        help="Zero every unfitted calibration constant — both efficiencies and the "
        "per-dispatch overhead: a hardware ceiling, not a prediction",
        rich_help_panel=PANEL_OUTPUT,
    ),
) -> None:
    """Head-to-head across chips, with the prefill crossover point."""
    try:
        chip_specs = [_chip_for(c.strip(), ideal) for c in chips.split(",")]
        model_specs = [load_model(m.strip()) for m in models.split(",")]
    except SpecLoadError as exc:
        console.print(f"[red]bwz:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    deployment = DeploymentSpec.model_validate(
        {
            "batch": batch,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "kv_context_tokens": context,
            "precision": {"weights": weights, "activations": weights, "kv_cache": weights},
        }
    )
    rows = head_to_head(model_specs, chip_specs, deployment)

    table = Table(
        title=(
            f"Head to head @ S={input_tokens}, batch {batch}, {weights.value}"
            + (" [ideal]" if ideal else "")
        ),
        box=box.SIMPLE,
    )
    table.add_column("model")
    table.add_column("chip")
    table.add_column("params", justify="right")
    table.add_column("resident", justify="right")
    table.add_column("TTFT", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("bound")
    table.add_column("util", justify="right")
    for row in rows:
        table.add_row(
            row.model_id,
            row.chip_id,
            format_quantity(row.parameter_count, "").strip(),
            f"{row.resident_fraction:.2%}",
            format_time(row.ttft_s) if row.ttft_s else "-",
            f"{row.tokens_per_s:.1f}" if row.tokens_per_s else "-",
            _colour_bound(row.bound),
            f"{row.utilization:.2%}",
        )
    console.print(table)

    if crossover and len(chip_specs) == 2:
        console.print("\n[bold]Prefill crossover[/bold]")
        for model_spec in model_specs:
            point = prefill_crossover(model_spec, chip_specs[0], chip_specs[1], deployment)
            if point.tokens is None:
                console.print(
                    f"  {model_spec.id}: no crossover in [{point.searched_lo}, "
                    f"{point.searched_hi}] — {point.faster_below} is faster throughout"
                )
            else:
                console.print(
                    f"  {model_spec.id}: S* = {point.tokens} tokens "
                    f"({point.faster_below} faster below, {point.faster_above} above)"
                )


@app.command(name="list")
def list_profiles(
    what: Catalog = typer.Argument(Catalog.ALL, help="chips, models, or all"),
) -> None:
    """List the bundled chip and model profiles.

    ``ridge`` is the arithmetic intensity in OP/byte at which the DRAM roofline
    meets the compute roofline: an operation below it is DRAM-bound.
    """
    try:
        if what in (Catalog.CHIPS, Catalog.ALL):
            console.print(_chip_table())
        if what is Catalog.ALL:
            console.print()
        if what in (Catalog.MODELS, Catalog.ALL):
            console.print(_model_table())
    except SpecLoadError as exc:
        console.print(f"[red]bwz:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
