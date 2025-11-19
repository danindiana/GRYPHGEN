"""Command-line interface for ShellGenie."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.table import Table

from shellgenie import __version__
from shellgenie.core import ShellGenieCore
from shellgenie.models import AppConfig, ModelBackend, ModelConfig, SecurityLevel
from shellgenie.utils import (
    check_dependencies,
    check_gpu_available,
    get_gpu_info,
    get_system_info,
    setup_logging,
)

console = Console()


def print_banner() -> None:
    """Print ShellGenie banner."""
    banner = """
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║      ███████╗██╗  ██╗███████╗██╗     ██╗         ║
    ║      ██╔════╝██║  ██║██╔════╝██║     ██║         ║
    ║      ███████╗███████║█████╗  ██║     ██║         ║
    ║      ╚════██║██╔══██║██╔══╝  ██║     ██║         ║
    ║      ███████║██║  ██║███████╗███████╗███████╗    ║
    ║      ╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝    ║
    ║                                                   ║
    ║         ██████╗ ███████╗███╗   ██╗██╗███████╗    ║
    ║        ██╔════╝ ██╔════╝████╗  ██║██║██╔════╝    ║
    ║        ██║  ███╗█████╗  ██╔██╗ ██║██║█████╗      ║
    ║        ██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝      ║
    ║        ╚██████╔╝███████╗██║ ╚████║██║███████╗    ║
    ║         ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝    ║
    ║                                                   ║
    ║           AI-Powered Bash Assistant              ║
    ║                 v{version}                      ║
    ╚═══════════════════════════════════════════════════╝
    """.format(version=__version__)
    console.print(banner, style="bold cyan")


@click.group()
@click.version_option(version=__version__, prog_name="ShellGenie")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """ShellGenie - AI-powered bash shell assistant.

    Transform natural language into bash commands with GPU-accelerated LLMs.
    """
    ctx.ensure_object(dict)
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(log_level)
    ctx.obj["DEBUG"] = debug


@cli.command()
@click.argument("prompt", nargs=-1, required=True)
@click.option("--execute", "-e", is_flag=True, help="Execute command immediately")
@click.option("--model", "-m", default="llama3.2", help="Model name to use")
@click.option("--backend", "-b", type=click.Choice(["ollama", "llama_cpp"]), default="ollama")
@click.option("--security", "-s", type=click.Choice(["strict", "moderate", "permissive", "disabled"]), default="moderate")
@click.pass_context
def run(
    ctx: click.Context,
    prompt: tuple,
    execute: bool,
    model: str,
    backend: str,
    security: str,
) -> None:
    """Generate and optionally execute a bash command.

    Examples:
        sg run "list all pdf files in current directory"
        sg run "show disk usage" --execute
        sg run "find large files over 1GB" -m llama3.1 -e
    """
    prompt_text = " ".join(prompt)

    # Create configurations
    model_config = ModelConfig(
        backend=ModelBackend(backend),
        model_name=model,
    )
    app_config = AppConfig(
        security_level=SecurityLevel(security),
        auto_execute=False,  # We handle execution manually
    )

    # Initialize core
    try:
        core = ShellGenieCore(model_config=model_config, app_config=app_config)

        # Process request
        console.print(f"\n[bold cyan]Processing:[/bold cyan] {prompt_text}\n")

        async def process():
            return await core.process_request(prompt_text)

        response, _ = asyncio.run(process())

        # Display response
        if not response.command:
            console.print("[bold red]Error:[/bold red] No command generated")
            if response.warnings:
                for warning in response.warnings:
                    console.print(f"[yellow]⚠[/yellow]  {warning}")
            sys.exit(1)

        # Show the generated command
        console.print(Panel(
            Syntax(response.command, "bash", theme="monokai", line_numbers=False),
            title="[bold green]Generated Command[/bold green]",
            border_style="green",
        ))

        # Show explanation if available
        if response.explanation:
            console.print(f"\n[dim]{response.explanation}[/dim]")

        # Show warnings
        if response.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in response.warnings:
                console.print(f"  [yellow]⚠[/yellow]  {warning}")

        # Show alternatives if available
        if response.alternatives:
            console.print("\n[bold cyan]Safer alternatives:[/bold cyan]")
            for alt in response.alternatives:
                console.print(f"  [cyan]→[/cyan]  {alt}")

        # Risk assessment
        risk_colors = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }
        risk_color = risk_colors.get(response.risk_level, "white")
        console.print(f"\n[bold]Risk Level:[/bold] [{risk_color}]{response.risk_level.upper()}[/{risk_color}]")

        # Execute if requested
        if execute or (response.risk_level == "low" and Confirm.ask("\nExecute this command?")):
            console.print("\n[bold cyan]Executing...[/bold cyan]\n")
            result = core.execute(response.command)

            if result.stdout:
                console.print(result.stdout)
            if result.stderr:
                console.print(f"[red]{result.stderr}[/red]")

            status = "[green]✓ Success[/green]" if result.success else "[red]✗ Failed[/red]"
            console.print(f"\n{status} (exit code: {result.return_code}, time: {result.execution_time:.2f}s)")

            sys.exit(result.return_code)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if ctx.obj.get("DEBUG"):
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@cli.command()
@click.option("--model", "-m", default="llama3.2", help="Model name to use")
@click.option("--backend", "-b", type=click.Choice(["ollama", "llama_cpp"]), default="ollama")
@click.option("--security", "-s", type=click.Choice(["strict", "moderate", "permissive"]), default="moderate")
def interactive(model: str, backend: str, security: str) -> None:
    """Start an interactive ShellGenie session.

    Enter natural language prompts and get bash commands in real-time.
    Type 'exit' or 'quit' to end the session.
    """
    print_banner()

    # Create configurations
    model_config = ModelConfig(
        backend=ModelBackend(backend),
        model_name=model,
    )
    app_config = AppConfig(
        security_level=SecurityLevel(security),
        auto_execute=False,
    )

    # Initialize core
    core = ShellGenieCore(model_config=model_config, app_config=app_config)

    console.print("\n[bold green]Interactive mode started.[/bold green]")
    console.print("Type your request in natural language, or 'exit' to quit.\n")

    while True:
        try:
            # Get prompt
            prompt_text = Prompt.ask("[bold cyan]ShellGenie[/bold cyan]")

            if prompt_text.lower() in ["exit", "quit", "q"]:
                console.print("\n[bold green]Goodbye![/bold green]\n")
                break

            if not prompt_text.strip():
                continue

            # Process request
            async def process():
                return await core.process_request(prompt_text)

            response, _ = asyncio.run(process())

            # Display response
            if not response.command:
                console.print("[bold red]Error:[/bold red] No command generated")
                if response.warnings:
                    for warning in response.warnings:
                        console.print(f"[yellow]⚠[/yellow]  {warning}")
                continue

            # Show the generated command
            console.print(Panel(
                Syntax(response.command, "bash", theme="monokai", line_numbers=False),
                title=f"[bold green]Command[/bold green] [dim](risk: {response.risk_level})[/dim]",
                border_style="green",
            ))

            # Show warnings
            if response.warnings:
                for warning in response.warnings:
                    console.print(f"  [yellow]⚠[/yellow]  {warning}")

            # Ask to execute
            if response.risk_level not in ["high", "critical"]:
                if Confirm.ask("Execute?", default=False):
                    result = core.execute(response.command)
                    if result.stdout:
                        console.print(result.stdout)
                    if result.stderr:
                        console.print(f"[red]{result.stderr}[/red]")

            console.print()  # Blank line

        except KeyboardInterrupt:
            console.print("\n\n[bold green]Goodbye![/bold green]\n")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}\n")


@cli.command()
def info() -> None:
    """Display system and configuration information."""
    print_banner()

    # System info
    sys_info = get_system_info()

    console.print("\n[bold cyan]System Information[/bold cyan]")
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("OS", f"{sys_info.os} {sys_info.kernel}")
    table.add_row("Shell", sys_info.shell)
    table.add_row("User", f"{sys_info.user}@{sys_info.hostname}")
    table.add_row("Working Directory", sys_info.cwd)

    console.print(table)

    # GPU info
    if sys_info.gpu_available:
        console.print("\n[bold green]GPU Information[/bold green]")
        gpu_info = get_gpu_info()

        for gpu in gpu_info.get("devices", []):
            gpu_table = Table(show_header=False, box=None)
            gpu_table.add_column("Key", style="green")
            gpu_table.add_column("Value", style="white")

            gpu_table.add_row("GPU", gpu["name"])
            gpu_table.add_row("Memory Total", gpu["memory_total"])
            gpu_table.add_row("Memory Used", gpu["memory_used"])
            gpu_table.add_row("Memory Free", gpu["memory_free"])
            gpu_table.add_row("Temperature", gpu["temperature"])
            gpu_table.add_row("GPU Utilization", gpu["gpu_utilization"])

            console.print(gpu_table)
            if sys_info.cuda_version:
                console.print(f"\n[bold]CUDA Version:[/bold] {sys_info.cuda_version}")
    else:
        console.print("\n[bold yellow]No GPU detected[/bold yellow]")

    # Dependencies
    console.print("\n[bold cyan]Dependencies[/bold cyan]")
    deps = check_dependencies()

    deps_table = Table(show_header=False, box=None)
    deps_table.add_column("Dependency", style="cyan")
    deps_table.add_column("Status", style="white")

    for dep, available in deps.items():
        status = "[green]✓ Available[/green]" if available else "[red]✗ Not available[/red]"
        deps_table.add_row(dep, status)

    console.print(deps_table)
    console.print()


@cli.command()
@click.option("--show-all", "-a", is_flag=True, help="Show all history entries")
@click.option("--clear", "-c", is_flag=True, help="Clear command history")
def history(show_all: bool, clear: bool) -> None:
    """View or manage command history."""
    app_config = AppConfig()

    if clear:
        if Confirm.ask("Clear all command history?"):
            from pathlib import Path
            history_path = Path(app_config.history_file).expanduser()
            if history_path.exists():
                history_path.unlink()
                console.print("[green]✓ History cleared[/green]")
            else:
                console.print("[yellow]No history file found[/yellow]")
        return

    from shellgenie.utils import load_command_history

    entries = load_command_history(app_config.history_file, 10000 if show_all else 20)

    if not entries:
        console.print("[yellow]No command history found[/yellow]")
        return

    console.print(f"\n[bold cyan]Command History[/bold cyan] [dim]({len(entries)} entries)[/dim]\n")

    for i, cmd in enumerate(entries[-20:] if not show_all else entries, 1):
        console.print(f"[cyan]{i:3d}[/cyan]  {cmd}")

    console.print()


@cli.command()
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table")
def stats(format: str) -> None:
    """Display usage statistics."""
    app_config = AppConfig()
    model_config = ModelConfig()

    core = ShellGenieCore(model_config=model_config, app_config=app_config)
    stats_data = core.get_stats()

    if format == "json":
        import json
        console.print(json.dumps(stats_data, indent=2))
    else:
        console.print("\n[bold cyan]ShellGenie Statistics[/bold cyan]\n")

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats_data.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)
        console.print()


@cli.command()
@click.argument("model_name")
@click.option("--backend", "-b", type=click.Choice(["ollama", "llama_cpp"]), default="ollama")
def test_model(model_name: str, backend: str) -> None:
    """Test if a model is working correctly.

    Example:
        sg test-model llama3.2
    """
    console.print(f"\n[bold cyan]Testing model:[/bold cyan] {model_name}\n")

    model_config = ModelConfig(
        backend=ModelBackend(backend),
        model_name=model_name,
    )
    app_config = AppConfig(security_level=SecurityLevel.PERMISSIVE)

    try:
        core = ShellGenieCore(model_config=model_config, app_config=app_config)

        test_prompt = "list files in current directory"
        console.print(f"[dim]Test prompt: {test_prompt}[/dim]\n")

        async def test():
            return await core.generate_command(
                from shellgenie.models import CommandRequest
                request = CommandRequest(prompt=test_prompt)
                return await core.generate_command(request)
            )

        response = asyncio.run(test())

        if response.command:
            console.print("[bold green]✓ Model is working![/bold green]\n")
            console.print(f"Generated command: [cyan]{response.command}[/cyan]\n")
        else:
            console.print("[bold red]✗ Model did not generate a command[/bold red]\n")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]✗ Model test failed:[/bold red] {str(e)}\n")
        sys.exit(1)


def main() -> None:
    """Main entry point for CLI."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Interrupted by user[/bold yellow]\n")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
