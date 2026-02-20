"""Entry point for experiment analysis tools.

Usage:
    python analyze_experiment.py --list
    python analyze_experiment.py <experiment> <tool> [args...]

Examples:
    python analyze_experiment.py mod_arithmetic metric_plot \
        --metrics training_metrics/loss training_metrics/val_acc \
        --layout overlay --smooth 0.9

    python analyze_experiment.py mod_arithmetic metric_plot \
        --metrics training_metrics/loss --strategies stride random \
        --format svg
"""

import sys
from pathlib import Path

from console import OLConsole, ConsoleConfig, ConsoleMode
from framework.cli_parser import OLArgumentParser
from analysis_tools import ToolRegistry, AnalysisContext
from analysis_tools.data_loader import load_experiment_data, load_experiment_config
from analysis_tools.metadata import MetricResolver
from analysis_tools.style import apply_style


def list_tools():
    """Print all registered analysis tools and exit."""
    console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
    console.print("\n[bold]Available analysis tools:[/bold]")
    for info in ToolRegistry.get_all_info():
        console.print(
            f"  [metric.value]{info['name']:25s}[/metric.value]  {info['description']}"
        )
    console.print()


def main():
    if '--list' in sys.argv:
        list_tools()
        return

    if len(sys.argv) < 3 or sys.argv[1].startswith('-'):
        is_help = '--help' in sys.argv or '-h' in sys.argv
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print()
        console.print("  [bold]Usage:[/bold]")
        console.print("    [metric.value]python analyze_experiment.py[/metric.value] [detail]<experiment> <tool> [args...][/detail]")
        console.print("    [metric.value]python analyze_experiment.py[/metric.value] [detail]--list[/detail]")
        console.print()
        try:
            list_tools()
        except Exception:
            console.print("  [detail](Could not discover tools)[/detail]")
        sys.exit(0 if is_help else 1)

    experiment_name = sys.argv[1]
    tool_name = sys.argv[2]

    # Resolve tool class
    try:
        tool_cls = ToolRegistry.get(tool_name)
    except ValueError as e:
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error(str(e))
        sys.exit(1)

    # Build parser with framework-level + tool-specific args
    parser = OLArgumentParser(
        experiment_name=f'{experiment_name} — {tool_name}',
        description=f'Run the {tool_name} analysis tool on {experiment_name}',
    )

    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument(
        '--output-dir', default='output',
        help='Base output directory (default: output)',
    )
    analysis_group.add_argument(
        '--strategies', nargs='+', default=None,
        help='Filter to specific strategies',
    )
    analysis_group.add_argument(
        '--range', nargs=2, type=int, metavar=('START', 'END'),
        dest='step_range',
        help='Step/epoch range filter (inclusive)',
    )
    analysis_group.add_argument(
        '--smooth', type=float, default=None,
        help='EMA smoothing weight (0-1)',
    )
    analysis_group.add_argument(
        '--format', choices=['png', 'svg'], default='png',
        help='Output image format (default: png)',
    )
    analysis_group.add_argument(
        '--dpi', type=int, default=300,
        help='Output image DPI (default: 300)',
    )
    analysis_group.add_argument(
        '--style', choices=['dark', 'paper'], default='dark',
        help='Plot style: dark (OLDarkTheme) or paper (publication-ready)',
    )
    analysis_group.add_argument(
        '--experiment-title', action='store_true', default=False,
        dest='experiment_title',
        help='Include experiment name in plot titles',
    )
    analysis_group.add_argument(
        '--layout', choices=['overlay', 'grid'], default='overlay',
        help='Layout mode: overlay (strategies share axes) or grid (subplots)',
    )
    analysis_group.add_argument(
        '--group-by', choices=['strategy', 'metric'], default='strategy',
        dest='group_by',
        help='In grid mode: what gets its own subplot (default: strategy)',
    )

    # Tool-specific args go in their own group
    tool_group = parser.add_argument_group(f'Tool Options ({tool_name})')
    tool_cls.add_args(tool_group)

    args = parser.parse_args(sys.argv[3:])

    console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))

    # Load data
    try:
        data = load_experiment_data(
            experiment_name,
            output_dir=args.output_dir,
            strategies=args.strategies,
        )
    except FileNotFoundError as e:
        console.print_error(str(e))
        sys.exit(1)

    # Apply step range filter
    if args.step_range and 'step' in data.columns:
        start, end = args.step_range
        data = data[(data['step'] >= start) & (data['step'] <= end)]
        if data.empty:
            console.print_error(
                f"No data in range [{start}, {end}]"
            )
            sys.exit(1)
        data = data.reset_index(drop=True)

    # Apply matplotlib style
    apply_style(args.style)

    # Load experiment config
    experiment_config = load_experiment_config(
        experiment_name, output_dir=args.output_dir,
    )

    # Build context
    strategies = sorted(data['strategy'].unique().tolist())
    tool_output_dir = (
        Path(args.output_dir) / experiment_name / 'analysis' / tool_name
    )

    resolver = MetricResolver()

    context = AnalysisContext(
        experiment_name=experiment_name,
        data=data,
        strategies=strategies,
        output_dir=tool_output_dir,
        args=args,
        experiment_config=experiment_config,
        resolver=resolver,
    )

    # Print header
    console.print()
    console.rule(f"[bold]{experiment_name}[/bold] — {tool_name}")
    console.print(f"  [label]Strategies:[/label] [strategy]{', '.join(strategies)}[/strategy]")
    console.print(f"  [label]Data points:[/label] [value.count]{len(data):,}[/value.count]")
    if experiment_config:
        model = experiment_config.get('model_size', '')
        if model:
            console.print(f"  [label]Model:[/label] [detail]{model}[/detail]")
    console.print()

    # Run tool
    tool = tool_cls()
    tool.run(context)


if __name__ == "__main__":
    main()
