"""Meta-learning step-based trainer.

Thin StepTrainer subclass that declares NESTED paradigm and
GLOBAL_GRADIENTS availability. The training loop is identical to
StepTrainer — the only difference is capability declarations, which
affect hook filtering via HookManager.set_capabilities().
"""

from __future__ import annotations

from ..capabilities import TrainingParadigm, GradientAvailability
from .step_trainer import StepTrainer


class MetaLearningStepTrainer(StepTrainer):
    """Step-based training for meta-learning paradigms (MAML, Reptile).

    Inherits the entire StepTrainer training loop. Overrides only the
    capability class attributes so that HookManager correctly filters
    hooks based on NESTED paradigm requirements.
    """

    paradigm = TrainingParadigm.NESTED
    gradient_availability = GradientAvailability.GLOBAL_GRADIENTS
