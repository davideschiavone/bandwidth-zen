"""Every command runs, on every kind of workload.

Not a check on the numbers — the golden tests do that — but on the plumbing.
Three CLI regressions in this session were shipped because nothing invoked the
commands: a missing lane in a display table, a flag that no longer existed, a
report path that raised. All of them would have failed here in a second.
"""

from __future__ import annotations

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
