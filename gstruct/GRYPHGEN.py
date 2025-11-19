#!/usr/bin/env python3
"""
GRYPHGEN - Advanced Grid Computing Framework
Main Entry Point

This is the main orchestrator that coordinates SYMORQ, SYMORG, and SYMAUG
components to provide a complete grid computing solution with LLM-based
orchestration and GPU acceleration.
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from SYMORQ.orchestration import Orchestrator
from SYMORG.scheduling import Scheduler
from SYMAUG.scripts.deployment import DeploymentManager
from utils.gpu_utils import check_gpu_availability, get_gpu_info
from config.settings import load_config

console = Console()


class GryphgenFramework:
    """Main GRYPHGEN framework coordinator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the GRYPHGEN framework.

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = load_config(config_path)
        self.orchestrator: Optional[Orchestrator] = None
        self.scheduler: Optional[Scheduler] = None
        self.deployment_manager: Optional[DeploymentManager] = None

        # Setup logging
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=self.config.get("log_level", "INFO"),
        )
        logger.add(
            "logs/gryphgen_{time}.log",
            rotation="500 MB",
            retention="10 days",
            level="DEBUG",
        )

    async def initialize(self):
        """Initialize all GRYPHGEN components."""
        logger.info("Initializing GRYPHGEN Framework...")

        # Check GPU availability
        gpu_available = check_gpu_availability()
        if gpu_available:
            gpu_info = get_gpu_info()
            logger.info(f"GPU detected: {gpu_info}")
            console.print(Panel(f"[green]✓[/green] GPU Available: {gpu_info}",
                              title="GPU Status"))
        else:
            logger.warning("No GPU detected. Running in CPU-only mode.")
            console.print(Panel("[yellow]⚠[/yellow] No GPU detected. CPU-only mode.",
                              title="GPU Status"))

        # Initialize components
        try:
            self.orchestrator = Orchestrator(self.config)
            await self.orchestrator.initialize()
            logger.info("✓ Orchestrator initialized")

            self.scheduler = Scheduler(self.config)
            await self.scheduler.initialize()
            logger.info("✓ Scheduler initialized")

            self.deployment_manager = DeploymentManager(self.config)
            await self.deployment_manager.initialize()
            logger.info("✓ Deployment Manager initialized")

            console.print(Panel("[green]✓[/green] All components initialized successfully",
                              title="Initialization Complete"))
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def start(self):
        """Start the GRYPHGEN framework."""
        logger.info("Starting GRYPHGEN Framework...")

        try:
            # Start all components
            await asyncio.gather(
                self.orchestrator.start(),
                self.scheduler.start(),
                self.deployment_manager.start(),
            )
        except Exception as e:
            logger.error(f"Error during framework execution: {e}")
            await self.shutdown()
            raise

    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down GRYPHGEN Framework...")

        try:
            shutdown_tasks = []
            if self.orchestrator:
                shutdown_tasks.append(self.orchestrator.shutdown())
            if self.scheduler:
                shutdown_tasks.append(self.scheduler.shutdown())
            if self.deployment_manager:
                shutdown_tasks.append(self.deployment_manager.shutdown())

            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            logger.info("✓ Shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def print_status(self):
        """Print current framework status."""
        table = Table(title="GRYPHGEN Framework Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")

        components = [
            ("Orchestrator", self.orchestrator),
            ("Scheduler", self.scheduler),
            ("Deployment Manager", self.deployment_manager),
        ]

        for name, component in components:
            if component:
                status = "Running" if hasattr(component, 'is_running') and component.is_running else "Initialized"
                details = component.get_status() if hasattr(component, 'get_status') else "N/A"
            else:
                status = "Not Initialized"
                details = "N/A"

            table.add_row(name, status, str(details))

        console.print(table)


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """GRYPHGEN - Advanced Grid Computing Framework with LLM Orchestration."""
    pass


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def start(config: Optional[str], debug: bool):
    """Start the GRYPHGEN framework."""
    console.print(Panel.fit(
        "[bold cyan]GRYPHGEN[/bold cyan]\n"
        "Grid Computing Framework v2.0.0\n"
        "LLM-based Orchestration • GPU Acceleration",
        border_style="cyan"
    ))

    async def _start():
        framework = GryphgenFramework(config)
        if debug:
            framework.config['log_level'] = 'DEBUG'

        try:
            await framework.initialize()
            await framework.start()
        except KeyboardInterrupt:
            console.print("\n[yellow]Received shutdown signal...[/yellow]")
        finally:
            await framework.shutdown()

    asyncio.run(_start())


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Path to configuration file')
def status(config: Optional[str]):
    """Show framework status."""
    framework = GryphgenFramework(config)
    framework.print_status()


@cli.command()
def info():
    """Display system and GPU information."""
    console.print(Panel.fit("[bold]System Information[/bold]", border_style="cyan"))

    gpu_available = check_gpu_availability()
    if gpu_available:
        gpu_info = get_gpu_info()
        console.print(f"\n[green]✓[/green] GPU: {gpu_info}")
    else:
        console.print("\n[yellow]⚠[/yellow] No GPU detected")

    # Display Python version
    console.print(f"\n[cyan]Python:[/cyan] {sys.version.split()[0]}")
    console.print(f"[cyan]Platform:[/cyan] {sys.platform}")


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
