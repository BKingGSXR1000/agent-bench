"""Milestone M0 command-line placeholder."""

from typing import Annotated

import typer

from agent_bench import __version__

app = typer.Typer(
    add_completion=False,
    help="Deterministic benchmark framework for coding-agent harnesses.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the Agent Bench version and exit."),
    ] = False,
) -> None:
    """Show help or version information; benchmark execution is not in M0."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


if __name__ == "__main__":
    app()
