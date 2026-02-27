"""Unified entry point for running experiments via the framework.

Usage:
    python run_experiment.py <experiment_name> [experiment-specific args]
    python run_experiment.py --config <experiment_config.json> [overrides]
    python run_experiment.py --interactive [--live]

Examples:
    python run_experiment.py presorted --live
    python run_experiment.py mod_arithmetic --strategy stride --epochs 1000
    python run_experiment.py phased_curriculum --model small --steps 10000
    python run_experiment.py guided_llm --target kepler --steps 5000

    # Rerun from saved config (with optional overrides)
    python run_experiment.py --config output/mod_arithmetic/stride/experiment_config.json
    python run_experiment.py --config output/mod_arithmetic/stride/experiment_config.json --epochs 2000

    # Interactive configuration
    python run_experiment.py --interactive
    python run_experiment.py --interactive --live

    # List available experiments
    python run_experiment.py --list

    # List/describe training hooks
    python run_experiment.py presorted --hooks-list
    python run_experiment.py presorted --hooks-describe norms
"""

import sys

import torch

from console import OLConsole, ConsoleConfig, ConsoleMode
from framework import ExperimentRegistry
from framework.trainers import StepTrainer, EpochTrainer
from framework.cli import (
    add_common_args, add_hook_args,
    handle_hook_inspection, build_hook_manager,
    OLArgumentParser,
)


def list_experiments():
    """Print all registered experiments and exit."""
    # Import experiments to trigger registration
    import experiments  # noqa: F401

    console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
    console.print("\n[bold]Available experiments:[/bold]")
    for name in ExperimentRegistry.list_all():
        runner_cls = ExperimentRegistry.get(name)
        doc = runner_cls.__doc__ or ""
        first_line = doc.strip().split('\n')[0] if doc.strip() else ""
        console.print(f"  [metric.value]{name:25s}[/metric.value]  {first_line}")
    console.print()


def _extract_config_flag(argv):
    """Extract --config path from argv.

    Returns (config_path, remaining_argv) where remaining_argv has the
    --config flag and its value removed. Returns (None, argv[1:]) if
    --config is not present.

    Handles both '--config path' and '--config=path' forms.
    """
    remaining = []
    config_path = None
    i = 1  # skip script name
    while i < len(argv):
        if argv[i] == '--config' and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif argv[i].startswith('--config='):
            config_path = argv[i].split('=', 1)[1]
            i += 1
        else:
            remaining.append(argv[i])
            i += 1
    return config_path, remaining


def _handle_config_mode(config_path, remaining_argv):
    """Load config from JSON, build parser with JSON defaults, parse overrides.

    1. Infer experiment name from config path directory structure
    2. Build full parser for that experiment
    3. Map JSON config fields to argparse defaults
    4. Parse remaining_argv (explicit CLI flags override JSON values)
    5. build_config(args) + patch unmapped fields from JSON

    Returns (experiment_name, runner_cls, config, args).
    """
    from pathlib import Path
    from framework.checkpoints import load_config_from_output

    config_file = Path(config_path)
    if not config_file.exists():
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    # Infer experiment name from path: output/{experiment}/{strategy}/experiment_config.json
    inferred_name = config_file.parent.parent.name

    # Check for optional positional experiment name in remaining args
    if remaining_argv and not remaining_argv[0].startswith('-'):
        experiment_name = remaining_argv[0]
        parse_argv = remaining_argv[1:]
        if experiment_name != inferred_name:
            console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
            console.print_error(
                f"Experiment name '{experiment_name}' doesn't match "
                f"config path (inferred: '{inferred_name}')"
            )
            sys.exit(1)
    else:
        experiment_name = inferred_name
        parse_argv = remaining_argv

    # Import experiments to trigger registration
    import experiments  # noqa: F401

    try:
        runner_cls = ExperimentRegistry.get(experiment_name)
    except KeyError as e:
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error(str(e))
        sys.exit(1)

    # Build parser with all argument groups
    parser = OLArgumentParser(
        experiment_name=experiment_name,
        description=f'Run the {experiment_name} experiment (from config)',
    )
    add_common_args(parser)
    add_hook_args(parser)
    runner_cls.add_args(parser)

    # Load full config from JSON
    json_config = load_config_from_output(config_path, runner_cls)

    # Map config fields to argparse dest names and set as defaults
    arg_dests = {a.dest for a in parser._actions}
    aliases = getattr(runner_cls, 'arg_aliases', {})
    defaults = {}
    for field_name in json_config.__dataclass_fields__:
        dest = aliases.get(field_name, field_name)
        if dest in arg_dests:
            defaults[dest] = getattr(json_config, field_name)
    parser.set_defaults(**defaults)

    # Parse CLI args — explicit flags override JSON defaults
    args = parser.parse_args(parse_argv)

    # Build config via normal build_config path (handles --quick etc.)
    config = runner_cls.build_config(args)

    # Patch unmapped fields: config fields with no CLI arg get JSON values
    for field_name in json_config.__dataclass_fields__:
        dest = aliases.get(field_name, field_name)
        if dest not in arg_dests:
            setattr(config, field_name, getattr(json_config, field_name))

    return experiment_name, runner_cls, config, args


def main():
    # Quick check for --list flag
    if '--list' in sys.argv:
        list_experiments()
        return

    is_interactive = '--interactive' in sys.argv
    config_path, config_remaining = _extract_config_flag(sys.argv)

    # Conflict checks
    if is_interactive and '--resume' in sys.argv:
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error("--resume cannot be used with --interactive")
        sys.exit(1)

    if config_path and is_interactive:
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error("--config cannot be used with --interactive")
        sys.exit(1)

    if config_path and '--resume' in sys.argv:
        console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
        console.print_error("--config cannot be used with --resume")
        sys.exit(1)

    if config_path:
        # Config file mode: load saved config, allow CLI overrides
        experiment_name, runner_cls, config, args = _handle_config_mode(
            config_path, config_remaining,
        )
        resume_info = None

    elif is_interactive:
        # Strip flags that won't be in experiment parser
        sys.argv.remove('--interactive')
        live_override = None
        if '--live' in sys.argv:
            sys.argv.remove('--live')
            live_override = True

        # Init console in NORMAL mode for prompts
        OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))

        from framework.cli import interactive_configure
        experiment_name, args = interactive_configure(live_override)

        runner_cls = ExperimentRegistry.get(experiment_name)
    else:
        # Existing flow: positional experiment name + argparse
        if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
            is_help = '--help' in sys.argv or '-h' in sys.argv
            console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
            console.print()
            console.print("  [bold]Usage:[/bold]")
            console.print("    [metric.value]python run_experiment.py[/metric.value] [detail]<experiment> [args...][/detail]")
            console.print("    [metric.value]python run_experiment.py[/metric.value] [detail]--config <experiment_config.json> [overrides][/detail]")
            console.print("    [metric.value]python run_experiment.py[/metric.value] [detail]--interactive [--live][/detail]")
            console.print("    [metric.value]python run_experiment.py[/metric.value] [detail]--list[/detail]")
            console.print()
            try:
                list_experiments()
            except Exception:
                console.print("  [detail](Could not discover experiments)[/detail]")
            sys.exit(0 if is_help else 1)

        experiment_name = sys.argv[1]

        # Import experiments to trigger registration
        import experiments  # noqa: F401

        try:
            runner_cls = ExperimentRegistry.get(experiment_name)
        except KeyError as e:
            console = OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
            console.print_error(str(e))
            sys.exit(1)

        # Build parser: common args + hook args + experiment-specific args
        parser = OLArgumentParser(
            experiment_name=experiment_name,
            description=f'Run the {experiment_name} experiment',
        )
        add_common_args(parser)
        add_hook_args(parser)
        runner_cls.add_args(parser)
        args = parser.parse_args(sys.argv[2:])  # Skip script name and experiment name

    # All paths converge here — select console mode with priority chain
    no_console = getattr(args, 'no_console_output', False)
    silent = getattr(args, 'silent', False)
    live_mode = getattr(args, 'live', False) and not no_console and not silent

    if no_console:
        mode = ConsoleMode.NULL
    elif silent:
        mode = ConsoleMode.SILENT
    elif live_mode:
        mode = ConsoleMode.LIVE
    else:
        mode = ConsoleMode.NORMAL
    console_config = ConsoleConfig(mode=mode, show_time=False)
    console = OLConsole(console_config)

    # Set matmul precision (before any model construction)
    torch.set_float32_matmul_precision(getattr(args, 'matmul_precision', 'highest'))

    # Handle hook inspection (--hooks-list, --hooks-describe)
    if handle_hook_inspection(args):
        return

    # Build config — from config file, checkpoint on resume, or CLI args
    if config_path:
        # Config already built by _handle_config_mode above
        pass
    elif getattr(args, 'resume', False):
        from framework.checkpoints import (
            check_resume_conflicts, detect_resume_state, load_config_from_output,
        )

        conflicts = check_resume_conflicts(sys.argv)
        if conflicts:
            console.print_error(
                f"Cannot use {', '.join(conflicts)} with --resume. "
                "Resume loads all training config from the saved experiment."
            )
            sys.exit(1)

        strategy = getattr(args, 'strategy', 'all')
        output_dir = getattr(args, 'output_dir', 'output')
        resume_info = detect_resume_state(experiment_name, strategy,
                                          output_dir, runner_cls)

        config = load_config_from_output(resume_info.config_path, runner_cls)
        config.experiment_name = experiment_name
        config.strategy = strategy
    else:
        config = runner_cls.build_config(args)
        resume_info = None

    if not config.experiment_name:
        config.experiment_name = experiment_name

    # Apply framework-level checkpoint settings from CLI
    if getattr(args, 'save_checkpoints', None) is not None:
        config.save_checkpoints = True
        if args.save_checkpoints > 0:
            config.checkpoint_every = args.save_checkpoints
    elif getattr(args, 'validate_checkpoints', None) is not None:
        config.validate_checkpoints = True
        if args.validate_checkpoints > 0:
            config.checkpoint_every = args.validate_checkpoints
    if getattr(args, 'no_determinism', False):
        config.no_determinism = True

    # Reject flag combinations that make checkpoint validation meaningless
    if config.validate_checkpoints:
        problems = []
        if config.no_determinism:
            problems.append(
                "--no-determinism disables deterministic algorithms, "
                "so checkpoint validation cannot produce bit-identical state"
            )
        if config.with_compile:
            problems.append(
                "--with-compile can introduce non-determinism from "
                "Triton/inductor kernel generation"
            )
        if problems:
            console.print_error(
                "--validate-checkpoints requires fully deterministic execution"
            )
            for p in problems:
                console.print(f"  [metric.degraded]•[/metric.degraded] {p}")
            sys.exit(1)

    hook_manager = build_hook_manager(
        args, config=config,
        wandb_group_prefix=experiment_name,
        loop_type=runner_cls.loop_type,
        hook_sets=getattr(runner_cls, 'hook_sets', None),
        live_metrics=getattr(runner_cls, 'live_metrics', None),
    )
    runner = runner_cls.build_runner(config, args)

    try:
        # Dispatch to the correct trainer based on runner.trainer_class or loop_type
        if runner.trainer_class is not None:
            trainer_cls = runner.trainer_class
        elif runner.loop_type == 'epoch':
            trainer_cls = EpochTrainer
        else:
            trainer_cls = StepTrainer
        trainer = trainer_cls(runner, hook_manager, resume=resume_info)
        if hook_manager:
            hook_manager.set_capabilities(trainer.get_capabilities())
        results = trainer.train()
    finally:
        if live_mode:
            console.end_live()

    # Print summary to console after live display exits
    if live_mode and results:
        from framework import display
        display.display_post_live_summary(results)

    return results


if __name__ == "__main__":
    main()
