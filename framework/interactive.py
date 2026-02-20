"""Interactive experiment configuration via Rich prompts.

Provides a prompt-based flow that walks the user through experiment setup,
then returns (experiment_name, args_namespace) for the normal training path.
"""

import argparse
from types import SimpleNamespace

from console import OLConsole
from framework.registry import ExperimentRegistry
from framework.cli import add_common_args, add_hook_args


def interactive_configure(live_override=None):
    """Run interactive configuration.

    Args:
        live_override: If True, skip the live-mode prompt and pre-set it.

    Returns:
        (experiment_name, args_namespace) — ready for build_config() et al.
    """
    console = OLConsole()
    # Import experiments to trigger registration
    import experiments  # noqa: F401

    # --- Step 1: Select experiment ---
    experiment_names = ExperimentRegistry.list_all()
    console.print("\n[bold]Available experiments:[/bold]")
    for name in experiment_names:
        runner_cls = ExperimentRegistry.get(name)
        doc = runner_cls.__doc__ or ""
        first_line = doc.strip().split('\n')[0] if doc.strip() else ""
        console.print(f"  [metric.value]{name}[/metric.value]  {first_line}")
    console.print()

    experiment_name = console.prompt(
        "Select experiment",
        default_value=experiment_names[0],
        choices=experiment_names,
    )
    runner_cls = ExperimentRegistry.get(experiment_name)

    # --- Step 2: Build parser and get defaults ---
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_hook_args(parser)
    runner_cls.add_args(parser)
    defaults_ns = parser.parse_args([])
    args_dict = vars(defaults_ns)

    # --- Step 3: Prompt experiment-specific args ---
    interactive_args = getattr(runner_cls, 'interactive_args', [])
    actions_by_dest = {a.dest: a for a in parser._actions if a.dest != 'help'}

    console.print(f"\n[bold]Configure {experiment_name}:[/bold]")
    for dest in interactive_args:
        action = actions_by_dest.get(dest)
        if action is not None:
            args_dict[dest] = _prompt_for_action(action, args_dict[dest])

    # --- Step 4: Prompt hook set ---
    hook_sets = getattr(runner_cls, 'hook_sets', None)
    if hook_sets:
        hook_choices = sorted(hook_sets.keys())
        console.print()
        hook_group = console.prompt(
            "Hook set",
            default_value='none',
            choices=hook_choices,
        )
        if hook_group != 'none':
            args_dict['with_hooks'] = hook_group

    # --- Step 5: Live display ---
    if live_override is not None:
        args_dict['live'] = live_override
    else:
        args_dict['live'] = console.confirm("Enable live display?", default_value=False)

    # --- Step 6: Advanced settings ---
    if console.confirm("Advanced settings?", default_value=False):
        _prompt_advanced(args_dict, actions_by_dest, skip=set(interactive_args))

    # --- Step 7: Summary ---
    _print_summary(experiment_name, runner_cls, args_dict)

    # --- Step 8: Confirm ---
    if not console.confirm("Start training?", default_value=True):
        console.print("[warning]Aborted.[/warning]")
        raise SystemExit(0)

    return experiment_name, SimpleNamespace(**args_dict)


def _prompt_for_action(action, current_value):
    """Inspect an argparse Action and generate an appropriate prompt."""
    console = OLConsole()
    label = action.dest.replace('_', ' ').title()
    help_text = action.help or ""

    # Choice-based
    if action.choices:
        return console.prompt(
            f"{label} ({help_text})" if help_text else label,
            default_value=str(current_value),
            choices=[str(c) for c in action.choices],
        )

    # Boolean (BooleanOptionalAction or store_true/store_false)
    if isinstance(action, argparse.BooleanOptionalAction) or \
       action.const is True or action.const is False:
        return console.confirm(
            f"{label}?" if not help_text else f"{label} ({help_text})?",
            default_value=bool(current_value),
        )

    # Integer type
    if action.type is int or isinstance(current_value, int):
        result = console.prompt(
            f"{label} ({help_text})" if help_text else label,
            default_value=str(current_value),
        )
        try:
            return int(result)
        except (ValueError, TypeError):
            return current_value

    # Fallback: string prompt
    return console.prompt(
        f"{label} ({help_text})" if help_text else label,
        default_value=str(current_value) if current_value is not None else None,
    )


ADVANCED_DESTS = [
    'seed', 'batch_size', 'snapshot_every', 'output_dir',
    'record_trajectory',
]


def _prompt_advanced(args_dict, actions_by_dest, skip):
    """Prompt for secondary settings not covered by interactive_args."""
    console = OLConsole()
    console.print("\n[bold]Advanced settings:[/bold]")
    for dest in ADVANCED_DESTS:
        if dest in skip or dest not in actions_by_dest:
            continue
        action = actions_by_dest[dest]
        if dest in args_dict:
            args_dict[dest] = _prompt_for_action(action, args_dict[dest])


def _print_summary(experiment_name, runner_cls, args_dict):
    """Render a configuration summary."""
    console = OLConsole()
    console.print("\n[bold]Configuration Summary:[/bold]")
    console.print(f"  [label]Experiment:[/label]  [metric.value]{experiment_name}[/metric.value]")

    # Show interactive args first
    interactive_args = getattr(runner_cls, 'interactive_args', [])
    for dest in interactive_args:
        if dest in args_dict:
            label = dest.replace('_', ' ').title()
            console.print(f"  [label]{label}:[/label]  [metric.value]{args_dict[dest]}[/metric.value]")

    # Show a few key settings
    for key in ('live', 'with_hooks', 'seed', 'batch_size', 'output_dir'):
        if key in args_dict and key not in interactive_args:
            val = args_dict[key]
            if val is not None and val is not False and val != 'none':
                label = key.replace('_', ' ').title()
                console.print(f"  [label]{label}:[/label]  [metric.value]{val}[/metric.value]")
    console.print()
