"""bwz command-line interface (typer)."""

from __future__ import annotations

from enum import StrEnum

import typer
from rich.console import Console
from rich.table import Table

import bwz
from bwz.spec import (
    CNNSpec,
    CustomSpec,
    DType,
    HardwareSpec,
    SpecLoadError,
    TransformerSpec,
    iter_chips,
    iter_models,
)
from bwz.spec.loaders import AnyModelSpec
from bwz.units import format_bandwidth, format_bytes, format_quantity

app = typer.Typer(
    name="bwz",
    help="bandwidth-zen: analytical performance model for neural-network inference.",
    no_args_is_help=True,
)
console = Console()


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
        p = model.params
        shape = f"L={p.layers} d={p.hidden} ffn={p.ffn_hidden} kv={p.effective_kv_heads}"
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
