"""bwz command-line interface (typer)."""

from __future__ import annotations

import typer

import bwz

app = typer.Typer(
    name="bwz",
    help="bandwidth-zen: analytical performance model for neural-network inference.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """bandwidth-zen CLI. Run a subcommand; see --help."""


@app.command()
def version() -> None:
    """Print the bwz version."""
    typer.echo(f"bwz {bwz.__version__}")


if __name__ == "__main__":
    app()
