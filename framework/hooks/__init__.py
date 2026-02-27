"""Hook system infrastructure: base types, registry, and manager.

Provides the hook lifecycle abstractions, registry for hook discovery,
and the HookManager that orchestrates hook execution during training.
"""

from .hook_point import MetricInfo, HookPoint, STEP_HOOK_POINTS, StepSchedule
from .training_hook import TrainingHook
from .intervention_hook import InterventionHook, DebugInterventionHook
from .registry import HookRegistry
from .manager import HookManager
from ..capabilities import HookNeeds

__all__ = [
    'MetricInfo',
    'HookPoint',
    'STEP_HOOK_POINTS',
    'StepSchedule',
    'TrainingHook',
    'InterventionHook',
    'DebugInterventionHook',
    'HookRegistry',
    'HookManager',
    'HookNeeds',
]

# Import training_hooks to trigger @HookRegistry.register on all hook modules.
# This must come after the exports above since hook modules import from here.
import training_hooks  # noqa: E402, F401
