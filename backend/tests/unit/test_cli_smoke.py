"""Every command runs, on every kind of workload.

Not a check on the numbers — the golden tests do that — but on the plumbing.
Three CLI regressions in this session were shipped because nothing invoked the
commands: a missing lane in a display table, a flag that no longer existed, a
report path that raised. All of them would have failed here in a second.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from bwz.cli import app

runner = CliRunner()

COMMANDS = [
    ["list"],
    ["version"],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb"],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb", "--ideal"],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb", "--no-pipeline"],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb", "-d", "int8"],
    [
        "matmul",
        "-M",
        "64",
        "-N",
        "64",
        "-K",
        "64",
        "-c",
        "a100_80gb",
        "-d",
        "int8",
        "--c",
        "int32",
    ],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb", "--a", "fp16", "--b", "int8"],
    ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "a100_80gb", "--json"],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "512",
        "-c",
        "metis_aipu",
        "-d",
        "int8",
        "--a-strategy",
        "stream",
    ],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "512",
        "-c",
        "metis_aipu",
        "-d",
        "int8",
        "--a-strategy",
        "whole",
    ],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "512",
        "-c",
        "metis_aipu",
        "-d",
        "int8",
        "--b-dataflow",
        "on-demand",
    ],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "512",
        "-c",
        "metis_aipu",
        "-d",
        "int8",
        "--b-dataflow",
        "persistent",
        "--iterations",
        "4",
    ],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "512",
        "-c",
        "metis_aipu",
        "-d",
        "int8",
        "--a-residency-tiles",
        "3",
        "--a-prefetch-depth",
        "2",
    ],
    ["encoder-layer", "-c", "a100_80gb"],
    ["encoder-layer", "-c", "a100_80gb", "--ideal", "--ffn", "32", "-S", "16"],
    ["encoder-layer", "-c", "chip_a", "--weights", "int8"],
    ["encoder-layer", "-c", "a100_80gb", "--json"],
    [
        "run",
        "-m",
        "single_layer_encoder_toy",
        "-c",
        "a100_80gb",
        "--input-tokens",
        "4",
        "--show-ops",
        "20",
        "--ideal",
    ],
    ["run", "-m", "llama3_8b", "-c", "a100_80gb", "--input-tokens", "128", "--output-tokens", "4"],
    [
        "run",
        "-m",
        "llama3_8b",
        "-c",
        "h100_sxm",
        "--batch",
        "1",
        "--input-tokens",
        "128",
        "--output-tokens",
        "8",
        "--weights",
        "fp16",
        "--attention",
        "flash2",
    ],
    ["run", "-m", "mobilenetv3", "-c", "a100_80gb"],
    ["run", "-m", "gemma3_4b", "-c", "chip_a", "--weights", "int8", "--input-tokens", "64"],
    [
        "run",
        "-m",
        "llama3_8b",
        "-c",
        "a100_80gb",
        "--input-tokens",
        "64",
        "--output-tokens",
        "4",
        "--a-strategy",
        "stream",
        "--b-dataflow",
        "on-demand",
    ],
    ["compare", "--chips", "chip_a,chip_b", "--models", "gemma3_4b"],
    # D53's two knobs, on both commands that take them: --stationarity naming
    # the chip's own (never a refusal, so exit code 0 stays meaningful) and
    # --split-k, which only os has anything to split but which every command
    # must accept and account for.
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "4096",
        "--chip",
        "a100_80gb",
        "--stationarity",
        "os",
        "--split-k",
        "4",
        "--ideal",
    ],
    [
        "matmul",
        "-M",
        "512",
        "-N",
        "512",
        "-K",
        "4096",
        "--chip",
        "metis_aipu",
        "-d",
        "int8",
        "--stationarity",
        "ws",
    ],
    [
        "run",
        "-m",
        "llama3_8b",
        "-c",
        "a100_80gb",
        "--input-tokens",
        "64",
        "--output-tokens",
        "4",
        "--split-k",
        "2",
    ],
]


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda a: " ".join(a)[:60])
def test_command_runs(argv: list[str]) -> None:
    result = runner.invoke(app, argv)
    assert result.exit_code == 0, f"{' '.join(argv)}\n{result.output}\n{result.exception}"


def test_an_unsupported_dtype_is_a_report_not_a_crash() -> None:
    """An infeasible configuration exits 2 with reasons, never a traceback
    (CLAUDE.md #8)."""
    result = runner.invoke(
        app, ["matmul", "-M", "64", "-N", "64", "-K", "64", "-c", "chip_a", "-d", "fp16"]
    )
    assert result.exit_code == 2
    assert "no compute unit for 'fp16'" in result.output


def test_the_result_width_flag_reaches_the_report() -> None:
    """D64: ``--out`` meant two things on this command and the dtype half lost.

    ``--out int32`` parsed as a *path* — the figure directory's flag of the same
    name — so the widening accumulator was silently dropped and C came back at
    the operand width. Every test that covered widening accumulators went
    through ``analyze`` rather than the CLI, which is how a documented flag
    stayed broken while the suite was green. This one goes through the CLI.
    """
    result = runner.invoke(
        app,
        [
            "matmul",
            "-M",
            "512",
            "-N",
            "512",
            "-K",
            "512",
            "-c",
            "a100_80gb",
            "-d",
            "int8",
            "--c",
            "int32",
            "--no-pipeline",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "512 x 512 x 4 B — widening accumulator" in result.output
    assert "1.05 MB" in result.output, "512*512*4 B, not the 262 kB an int8 result would be"


def test_emit_and_the_figures_share_one_destination(tmp_path: Path) -> None:
    """D65: ``--emit`` used to carry a PATH of its own.

    So ``--emit X --out Y`` was two destinations for one run's artifacts, and
    ``--emit --timeline`` — the natural thing to type — consumed ``--timeline``
    as that path and wrote the program to a file literally named ``--timeline``.
    Both now land in ``--out``.
    """
    result = runner.invoke(
        app,
        [
            "matmul",
            "-M",
            "512",
            "-N",
            "512",
            "-K",
            "512",
            "-c",
            "a100_80gb",
            "--stationarity",
            "is",
            "--ideal",
            "--emit",
            "--timeline",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "matmul-a100_80gb-fp16-is.py",
        "timeline-a100_80gb-fp16-is.html",
    ]


def test_a_page_is_named_for_the_decomposition_it_draws(tmp_path: Path) -> None:
    """Two stationarities of one shape must not overwrite each other (D65).

    The page carries the same ``-<stationarity>[-splitk<N>]`` the emitted
    program does, so a directory of them reads as the comparison it is. Before
    this, ``--stationarity is`` landed on top of ``os``'s page and the only way
    to tell them apart was to open one.
    """
    for flow in ("os", "is", "ws"):
        result = runner.invoke(
            app,
            [
                "matmul",
                "-M",
                "512",
                "-N",
                "512",
                "-K",
                "512",
                "-c",
                "a100_80gb",
                "--stationarity",
                flow,
                "--ideal",
                "--timeline",
                "--out",
                str(tmp_path),
                "-q",
            ],
        )
        assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "matmul",
            "-M",
            "512",
            "-N",
            "512",
            "-K",
            "512",
            "-c",
            "a100_80gb",
            "--split-k",
            "4",
            "--ideal",
            "--timeline",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "timeline-a100_80gb-fp16-is.html",
        "timeline-a100_80gb-fp16-os-splitk4.html",
        "timeline-a100_80gb-fp16-os.html",
        "timeline-a100_80gb-fp16-ws.html",
    ]


def test_a_comparison_of_different_dataflows_names_neither(tmp_path: Path) -> None:
    """A100 runs ``os`` natively and Metis ``ws``, so one page draws two
    decompositions and there is no single one to put in its name — the same
    condition that leaves the stationarity banner blank (D65)."""
    result = runner.invoke(
        app,
        [
            "matmul",
            "-M",
            "512",
            "-N",
            "512",
            "-K",
            "512",
            "-c",
            "a100_80gb",
            "-d",
            "int8",
            "--compare-with",
            "metis_aipu",
            "--ideal",
            "--timeline",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    assert [p.name for p in tmp_path.iterdir()] == [
        "timeline-compare-a100_80gb-vs-metis_aipu-int8.html"
    ]


def test_emit_stdout_prints_the_program_and_nothing_else() -> None:
    """The ``| python -`` pipeline: a single stray line would be a syntax error."""
    result = runner.invoke(
        app,
        [
            "matmul",
            "-M",
            "64",
            "-N",
            "64",
            "-K",
            "128",
            "-c",
            "a100_80gb",
            "--stationarity",
            "ws",
            "--emit-stdout",
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("#!/usr/bin/env python3")
    assert result.output.rstrip().endswith("raise SystemExit(main())")


def test_attention_draws_its_timeline(tmp_path: Path) -> None:
    """``bwz attention --timeline`` writes the page named for the plan (D71)."""
    result = runner.invoke(
        app,
        [
            "attention",
            "-c",
            "a100_80gb",
            "-S",
            "64",
            "--head-dim",
            "16",
            "--heads",
            "2",
            "--br",
            "16",
            "--bc",
            "16",
            "--timeline",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "timeline-a100_80gb-flash-S64-d16-h2-br16-bc16-fp16.html").exists()


def test_attention_compares_two_chips_on_one_page(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "attention",
            "-c",
            "a100_80gb",
            "-S",
            "64",
            "--head-dim",
            "16",
            "--timeline",
            "--compare-with",
            "h100_sxm",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "timeline-compare-a100_80gb-vs-h100_sxm-flash-S64-d16-h1-fp16.html").exists()


def test_attention_animates(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "attention",
            "-c",
            "a100_80gb",
            "-S",
            "64",
            "--head-dim",
            "16",
            "--heads",
            "2",
            "--br",
            "16",
            "--bc",
            "16",
            "--animate",
            "--out",
            str(tmp_path),
            "-q",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "animate-a100_80gb-flash-S64-d16-h2-br16-bc16-fp16.html").exists()


def test_attention_refuses_to_animate_two_chips(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "attention",
            "-c",
            "a100_80gb",
            "-S",
            "64",
            "--head-dim",
            "16",
            "--animate",
            "--compare-with",
            "h100_sxm",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "--animate plays back one chip" in result.output
