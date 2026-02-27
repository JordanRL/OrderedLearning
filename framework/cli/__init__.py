"""CLI argument helpers, parser, and interactive configuration.

Provides shared argument groups, Rich-formatted argument parser,
hook inspection/building, and interactive experiment setup.
"""

from .cli import (
    add_common_args, add_hook_args, add_eval_target_args,
    handle_hook_inspection, build_hook_manager,
)
from .cli_parser import OLArgumentParser
from .interactive import interactive_configure

__all__ = [
    'add_common_args', 'add_hook_args', 'add_eval_target_args',
    'handle_hook_inspection', 'build_hook_manager',
    'OLArgumentParser',
    'interactive_configure',
]
