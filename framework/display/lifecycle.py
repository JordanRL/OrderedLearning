"""Experiment lifecycle display functions.

Banner, condition header, training start, and resume info panels
shown at experiment setup time.
"""

from rich.align import Align
from rich.table import Table
from rich.panel import Panel
from rich.rule import Rule
from rich import box

from console import OLConsole


def display_experiment_banner(title, description=None,
                              targets=None, extra_lines=None):
    """Primary panel with experiment overview. Called once at start."""
    console = OLConsole()
    lines = []
    if description:
        lines.append(description)
    if targets:
        lines.append("")
        for t in targets:
            lines.append(f"  [target]{t}[/target]")
    if extra_lines:
        lines.append("")
        for line in extra_lines:
            lines.append(f"  [detail]{line}[/detail]")

    content = "\n".join(lines) if lines else ""
    console.print(Panel(
        content,
        title=f"[bold]{title}[/bold]",
        border_style="panel.primary",
    ))


def display_condition_header(strategy_name, settings=None):
    """Yellow rule + info table when starting a new strategy/condition."""
    console = OLConsole()
    console.print()
    console.print(Rule(f"[bold][strategy]{strategy_name}[/strategy][/bold]",
                       style="panel.attention"))
    if settings:
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        table.add_column("Setting", style="label")
        table.add_column("Value", style="metric.value")
        for key, value in settings.items():
            table.add_row(key, str(value))
        console.print(Align.center(table))


def display_training_start(total, batch_size, extra_rows=None):
    """Green rule + settings table before training begins."""
    console = OLConsole()
    console.print(Rule("[bold][panel.success]Training[/panel.success][/bold]", style="panel.success"))
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Setting", style="label")
    table.add_column("Value", style="metric.value")
    table.add_row("Total", f"{total:,}")
    table.add_row("Batch size", str(batch_size))
    if extra_rows:
        for key, value in extra_rows.items():
            table.add_row(key, str(value))
    console.print(table)


def display_resume_info(step_or_epoch, checkpoint_path, completed_strategies,
                        counter_label="step"):
    """Display resume information when continuing from a checkpoint."""
    console = OLConsole()
    lines = [
        f"[label]Resuming from checkpoint at {counter_label} "
        f"{step_or_epoch:,}[/label]",
        f"[detail]{checkpoint_path}[/detail]",
    ]
    if completed_strategies:
        names = ', '.join(completed_strategies)
        lines.append(
            f"[label]Skipping completed strategies:[/label] [detail]{names}[/detail]"
        )
    console.print(Panel(
        "\n".join(lines),
        title="[bold][panel.info]Resume[/panel.info][/bold]",
        border_style="panel.info",
    ))
