"""Custom ArgumentParser with Rich-formatted help and error output.

Replaces argparse's default plain-text help formatter and stderr error
messages with Rich-rendered output through OLConsole, so all CLI output
uses the same styling system as the rest of the framework.
"""

import argparse
import sys

from rich.table import Table


class OLArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that renders help and errors through OLConsole.

    All CLI output goes through the project's Rich console system:
    - Help (--help) renders with grouped, styled argument display
    - Errors render as clean single-line messages via print_error()
    - No usage dump on error (just the error message)
    - Exit codes: 0 for help, 1 for errors
    """

    # Map default argparse group titles to display titles.
    # None means skip the group entirely.
    _GROUP_TITLES = {
        'positional arguments': None,
        'options': 'Experiment Options',           # Python 3.10+
        'optional arguments': 'Experiment Options', # Python <3.10
    }

    def __init__(self, experiment_name=None, **kwargs):
        # Disable argparse's built-in help so it flows through our overrides
        kwargs.setdefault('add_help', False)
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        # Re-add --help; the default _HelpAction calls print_help() + exit(0)
        self.add_argument(
            '-h', '--help',
            action='help',
            default=argparse.SUPPRESS,
            help='Show this help message and exit',
        )

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def error(self, message):
        """Print a clean error message and exit."""
        console = self._ensure_console()
        console.print_error(message)
        raise SystemExit(1)

    def exit(self, status=0, message=None):
        """Exit, optionally printing a message."""
        if message:
            console = self._ensure_console()
            stripped = message.strip()
            if status != 0:
                console.print_error(stripped)
            else:
                console.print(stripped)
        raise SystemExit(status)

    def print_help(self, file=None):
        """Render Rich-formatted help output."""
        self._format_help_rich()

    def print_usage(self, file=None):
        """Suppress standalone usage printing (errors don't need it)."""
        pass

    # ------------------------------------------------------------------
    # Rich help rendering
    # ------------------------------------------------------------------

    def _format_help_rich(self):
        """Build and print Rich-formatted help grouped by argument section."""
        console = self._ensure_console()

        # Title
        title = self.experiment_name or 'OrderedLearning'
        console.rule(title, style='rule.line')

        # Description
        if self.description:
            console.print(f'  {self.description}')
        console.print()

        # Render each argument group
        for group in self._action_groups:
            self._format_help_group(console, group)

    def _format_help_group(self, console, group):
        """Render a single argument group as aligned Rich columns."""
        # Filter out the help action itself
        actions = [a for a in group._group_actions
                   if not isinstance(a, argparse._HelpAction)]
        if not actions:
            return

        # Determine display title — skip groups mapped to None
        raw_title = group.title or 'Options'
        display_title = self._GROUP_TITLES.get(raw_title, raw_title)
        if display_title is None:
            return

        console.print(f'  [bold]{display_title.upper()}[/bold]')

        table = Table(
            box=None,
            show_header=False,
            padding=(0, 2),
            pad_edge=False,
        )
        table.add_column('flags', no_wrap=True)
        table.add_column('help')

        for action in actions:
            flags_str = self._format_action_flags(action)
            help_text = action.help or ''
            # Substitute %(default)s if present
            if '%(default)s' in help_text:
                help_text = help_text % {'default': action.default}
            table.add_row(
                f'    [metric.value]{flags_str}[/metric.value]',
                f'[detail]{help_text}[/detail]',
            )

        console.print(table)
        console.print()

    @staticmethod
    def _format_action_flags(action):
        """Format an action's flags + metavar into a display string."""
        parts = []

        if action.option_strings:
            parts.append(', '.join(action.option_strings))
        else:
            parts.append(action.dest)

        # Append type/choices hint
        if action.choices:
            parts.append('{' + ','.join(str(c) for c in action.choices) + '}')
        elif isinstance(action, (argparse._StoreTrueAction,
                                 argparse._StoreFalseAction)):
            pass  # boolean flags have no metavar
        elif isinstance(action, argparse.BooleanOptionalAction):
            pass  # --flag/--no-flag
        elif action.nargs in ('*', ):
            meta = action.metavar or action.dest.upper()
            if isinstance(meta, tuple):
                parts.append(' '.join(meta))
            else:
                parts.append(f'[{meta} ...]')
        elif action.nargs == '+':
            meta = action.metavar or action.dest.upper()
            parts.append(f'{meta} [{meta} ...]')
        elif action.metavar:
            if isinstance(action.metavar, tuple):
                parts.append(' '.join(action.metavar))
            else:
                parts.append(action.metavar)
        elif action.type and action.type is not type(None):
            parts.append(action.type.__name__.upper())

        return ' '.join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_console(self):
        """Get or create a NORMAL-mode console for help/error output.

        On help/error paths the program exits before run_experiment.py
        re-initializes the console with the user-requested mode, so
        creating a NORMAL console here is safe.
        """
        from console import OLConsole, ConsoleConfig, ConsoleMode
        return OLConsole(ConsoleConfig(mode=ConsoleMode.NORMAL, show_time=False))
