"""
Training hook system for live analysis during model training.

Hooks compute scalar metrics from live training data at configurable
lifecycle points (pre/post epoch, pre/post step, snapshot). This avoids
the need to save full parameter/gradient tensors for post-hoc analysis.

Usage:
    from training_hooks import HookManager, HookRegistry, ConsoleSink, CSVSink

    # Create a hook manager with desired hooks
    manager = HookManager(
        hook_names=['norms', 'consecutive', 'fourier'],
        sinks=[ConsoleSink(), CSVSink('metrics.csv')],
        offload_state=False,
    )

    # Pass to training loop
    do_train(..., hook_manager=manager)
"""

# Base classes and infrastructure
from .base import (
    MetricInfo,
    HookPoint,
    StepSchedule,
    TrainingHook,
    InterventionHook,
    DebugInterventionHook,
    HookRegistry,
    HookManager,
)

# Context objects
from .contexts import (
    RunDataContext,
    ModelDataContext,
    CheckpointProfiler,
)

# Metric sinks
from .sinks import (
    MetricSink,
    ConsoleSink,
    CSVSink,
    JSONLSink,
    WandbSink,
)

# Gradient accumulation helpers
from .grad_accumulator import (
    create_accumulator,
    accumulate,
    finalize,
)

# Import all hooks to register them with HookRegistry
from . import norms_hook
from . import token_gradient_hook
from . import fourier_hook
from . import attention_hook
from . import consecutive_hook
from . import phases_hook
from . import variance_hook
from . import subspace_gradient_info_hook
from . import counterfactual_hook
from . import counterfactual_validator_hook
from . import weight_tracking_hook
from . import hessian_hook
from . import gradient_projection_hook
from . import training_metrics_hook
from . import parameter_delta_hook
from . import path_length_hook
from . import batch_dynamics_hook
from . import adam_dynamics_hook
from . import training_diagnostics_hook

__all__ = [
    # Base classes
    'MetricInfo',
    'HookPoint',
    'StepSchedule',
    'TrainingHook',
    'InterventionHook',
    'DebugInterventionHook',
    'HookRegistry',
    'HookManager',
    # Contexts
    'RunDataContext',
    'ModelDataContext',
    'CheckpointProfiler',
    # Sinks
    'MetricSink',
    'ConsoleSink',
    'CSVSink',
    'JSONLSink',
    'WandbSink',
    # Grad accumulation
    'create_accumulator',
    'accumulate',
    'finalize',
]
