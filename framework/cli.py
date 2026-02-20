"""Shared CLI argument helpers for experiment entry points.

Provides functions to add common argument groups to argparse parsers
and build framework components from parsed args.

Extracted from duplicated CLI patterns across all four experiment files.
"""

import sys
from datetime import datetime
from dataclasses import asdict

from console import OLConsole


def add_common_args(parser, defaults=None):
    """Add common arguments shared by all experiments.

    Args:
        parser: argparse.ArgumentParser instance.
        defaults: Optional dict or dataclass with default values.
    """
    if defaults is not None and hasattr(defaults, '__dataclass_fields__'):
        d = defaults
    else:
        d = None

    group = parser.add_argument_group('Common Options')
    group.add_argument('--seed', type=int, default=getattr(d, 'seed', 199),
                       help=f"Random seed (default: {getattr(d, 'seed', 199)})")
    group.add_argument('--output-dir', type=str, default=getattr(d, 'output_dir', 'output'),
                       help='Output directory for experiment data')
    group.add_argument('--record-trajectory', action='store_true',
                       default=getattr(d, 'record_trajectory', False),
                       help='Record training trajectory')
    group.add_argument('--live', action='store_true', default=False,
                       help='Use LIVE console mode with full-screen UI '
                            '(overridden by --silent and --no-console-output)')
    group.add_argument('--silent', action='store_true', default=False,
                       help='Show only the progress bar; suppress all other output')
    group.add_argument('--no-console-output', action='store_true', default=False,
                       help='Suppress all console output')
    group.add_argument('--no-compile', action='store_true', default=False,
                       help='Disable torch.compile (useful for debugging or unsupported environments)')
    group.add_argument('--no-determinism', action='store_true', default=False,
                       help='Disable torch deterministic algorithms '
                            '(seeds still set, but allows non-deterministic CUDA ops)')
    group.add_argument('--resume', action='store_true', default=False,
                       help='Resume training from the most recent checkpoint in the output directory')

    ckpt_group = parser.add_argument_group('Checkpoint Options')
    ckpt_mutex = ckpt_group.add_mutually_exclusive_group()
    ckpt_mutex.add_argument('--save-checkpoints', nargs='?', type=int, const=0,
                            default=None, metavar='INTERVAL',
                            help='Save checkpoints at default interval, or every INTERVAL steps/epochs')
    ckpt_mutex.add_argument('--validate-checkpoints', nargs='?', type=int, const=0,
                            default=None, metavar='INTERVAL',
                            help='Validate training state against existing checkpoints for bit-identity')


def add_hook_args(parser):
    """Add training hook arguments to parser."""
    group = parser.add_argument_group('Hook Options')
    group.add_argument('--with-hooks', type=str, default=None, metavar='GROUP',
                       help='Enable experiment-defined hook group: '
                            'none, minimal, observers, interventions, full')
    group.add_argument('--hooks', nargs='*', default=None,
                       help='Training hooks to enable (e.g., norms consecutive). '
                            'Use "all" for all hooks, "observers" for non-intervention hooks only, '
                            '"with_debug" to include debug hooks. Additive with --with-hooks.')
    group.add_argument('--hook-offload-state', action='store_true', default=False,
                       help='Offload hook state tensors to CPU between calls')
    group.add_argument('--hook-csv', action='store_true', default=False,
                       help='Write hook metrics CSV to experiment output directory')
    group.add_argument('--hook-jsonl', action='store_true', default=False,
                       help='Write hook metrics JSONL to experiment output directory')
    group.add_argument('--hook-wandb', type=str, default=None, metavar='PROJECT',
                       help='Log hook metrics to Weights & Biases (provide project name)')
    group.add_argument('--hook-config', nargs='*', default=None,
                       help='Per-hook config: hook_name.param=value '
                            '(e.g., gradient_projection.reference_path=model.pt hessian.epsilon=0.01)')
    group.add_argument('--profile-hooks', action='store_true', default=False,
                       help='Profile hook checkpoint/compute operations')
    group.add_argument('--hooks-list', action='store_true', default=False,
                       help='List available training hooks and exit')
    group.add_argument('--hooks-describe', nargs='*', default=None, metavar='HOOK',
                       help='Describe metrics for hooks and exit. No args = all hooks.')


def add_eval_target_args(parser):
    """Add evaluation target arguments to parser."""
    group = parser.add_argument_group('Evaluation Targets')
    group.add_argument('--trigger', type=str, default=None,
                       help='Single eval target trigger (overrides defaults)')
    group.add_argument('--completion', type=str, default=None,
                       help='Single eval target completion (overrides defaults)')
    group.add_argument('--targets-file', type=str, default=None,
                       help='JSON file with eval targets: [{"trigger": "...", "completion": "...", "label": "..."}]')


def handle_hook_inspection(args):
    """Handle --hooks-list and --hooks-describe flags.

    Returns True if hook inspection was handled (caller should exit).
    Returns False if no inspection was requested.
    """
    if not (args.hooks_list or args.hooks_describe is not None):
        return False

    from training_hooks import HookRegistry
    console = OLConsole()

    if args.hooks_list:
        console.print("\n[bold]Available Training Hooks:[/bold]")
        for info in HookRegistry.get_all_info():
            kind = "[hook.type.intervention]intervention[/hook.type.intervention]" if info['is_intervention'] else "[hook.type.observer]observer[/hook.type.observer]"
            console.print(f"  [hook.name]{info['name']:20s}[/hook.name] {kind}  {info['description']}")
        console.print()

    if args.hooks_describe is not None:
        names = args.hooks_describe if args.hooks_describe else HookRegistry.list_all()
        for name in names:
            hook = HookRegistry.get(name)()
            hook.print_metric_descriptions()

    return True


def build_hook_manager(args, config=None, wandb_group_prefix="experiment",
                       loop_type=None, hook_sets=None, live_metrics=None):
    """Build a HookManager from parsed CLI args.

    Args:
        args: Parsed argparse namespace with hook args.
        config: Optional config dataclass for W&B logging.
        wandb_group_prefix: Prefix for W&B group name.
        loop_type: Loop type name ('step' or 'epoch') for per-loop hook
                   point resolution.
        hook_sets: Experiment-defined hook groups (from runner_cls.hook_sets).
                   Used with --with-hooks to select curated hook sets.
        live_metrics: Optional dict mapping display labels to metric keys
                      for the LIVE mode sidebar column.

    Returns:
        HookManager instance, or None if no hooks requested.
    """
    console = OLConsole()

    # Resolve --with-hooks group names into base hook list
    with_hooks_group = getattr(args, 'with_hooks', None)
    has_with_hooks = with_hooks_group is not None
    has_hooks = args.hooks is not None

    if not has_with_hooks and not has_hooks:
        return None

    base_names = []
    if has_with_hooks:
        if hook_sets is None:
            console.print_error("--with-hooks requires an experiment with hook_sets defined")
            sys.exit(1)
        if with_hooks_group not in hook_sets:
            valid = ', '.join(sorted(hook_sets.keys()))
            console.print_error(f"Unknown hook group '{with_hooks_group}'. Valid: {valid}")
            sys.exit(1)
        base_names = list(hook_sets[with_hooks_group])

    if not has_hooks and not base_names:
        return None

    from training_hooks import HookManager, HookRegistry, ConsoleSink, CSVSink, JSONLSink, WandbSink

    # Resolve hook names: start with --with-hooks base, add --hooks on top
    hook_names = list(base_names)
    if has_hooks:
        for name in args.hooks:
            if name == 'with_debug':
                hook_names.extend(HookRegistry.list_debug())
            elif name == 'all':
                hook_names.extend(HookRegistry.list_all())
            elif name == 'observers':
                hook_names.extend(HookRegistry.list_observers())
            else:
                hook_names.append(name)

    # training_metrics is always included when hooks are active — it
    # provides core loss/lr/accuracy metrics to sinks
    if 'training_metrics' not in hook_names:
        hook_names.insert(0, 'training_metrics')

    # Deduplicate preserving order
    seen = set()
    hook_names = [n for n in hook_names if not (n in seen or seen.add(n))]

    # Parse per-hook config
    hook_config = {}
    if args.hook_config:
        for item in args.hook_config:
            if '=' not in item or '.' not in item.split('=', 1)[0]:
                console.print_error(f"Invalid --hook-config format: '{item}' "
                                    f"(expected hook_name.param=value)")
                sys.exit(1)
            key, value = item.split('=', 1)
            hook_name, param = key.split('.', 1)
            # Auto-convert numeric values
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # keep as string
            hook_config.setdefault(hook_name, {})[param] = value

    # Build sinks
    sinks = [ConsoleSink(live_metrics=live_metrics)]
    if args.hook_csv:
        sinks.append(CSVSink(
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
        ))
    if args.hook_jsonl:
        sinks.append(JSONLSink(
            output_dir=config.output_dir,
            experiment_name=config.experiment_name,
        ))
    if args.hook_wandb:
        wandb_group = f"{wandb_group_prefix}-{datetime.now():%Y%m%d_%H%M%S}"
        config_dict = asdict(config) if config is not None and hasattr(config, '__dataclass_fields__') else {}
        sinks.append(WandbSink(project=args.hook_wandb, group=wandb_group, config=config_dict))

    hook_manager = HookManager(
        hook_names=hook_names,
        sinks=sinks,
        offload_state=getattr(args, 'hook_offload_state', False),
        hook_config=hook_config,
        profile_hooks=getattr(args, 'profile_hooks', False),
        loop_type=loop_type,
    )
    console.print(f"[label]Hooks enabled:[/label] [metric.value]{', '.join(hook_names)}[/metric.value]")

    return hook_manager
