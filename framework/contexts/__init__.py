"""Training context objects passed to hooks at lifecycle points.

- RunContext: frozen, run-agnostic lifecycle context (always passed to hooks)
- BackpropModelState, BackpropGradientState: paradigm-specific state objects
- AdversarialModelState, AdversarialGradientState: GAN state objects
- RLModelState, RLGradientState: RL state objects
- PredictiveCodingModelState, PredictiveCodingGradientState: PC state objects
- EvalMetrics, BatchState: generic state objects
- BackpropInterventionContext: mutable SAPI for intervention hooks (backprop)
- CheckpointProfiler: timing utility for hook and checkpoint operations
"""

from .profiler import CheckpointProfiler
from .run_context import RunContext
from .model_state import BackpropModelState
from .gradient_state import BackpropGradientState
from .adversarial_state import AdversarialModelState, AdversarialGradientState
from .rl_state import RLModelState, RLGradientState
from .pc_state import PredictiveCodingModelState, PredictiveCodingGradientState
from .meta_state import MetaLearningModelState, MetaLearningGradientState
from .evolutionary_state import EvolutionaryModelState, EvolutionaryGradientState
from .eval_metrics import EvalMetrics
from .batch_state import BatchState
from .model_context import BackpropInterventionContext

__all__ = [
    'CheckpointProfiler',
    'RunContext',
    'BackpropModelState',
    'BackpropGradientState',
    'AdversarialModelState', 'AdversarialGradientState',
    'RLModelState', 'RLGradientState',
    'PredictiveCodingModelState', 'PredictiveCodingGradientState',
    'MetaLearningModelState', 'MetaLearningGradientState',
    'EvolutionaryModelState', 'EvolutionaryGradientState',
    'EvalMetrics',
    'BatchState',
    'BackpropInterventionContext',
]
