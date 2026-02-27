"""Local learning step-based trainer.

Thin StepTrainer subclass that declares LOCAL_LEARNING paradigm and
LOCAL_GRADIENTS availability. The training loop is identical to
StepTrainer — the only difference is capability declarations, which
affect hook filtering via HookManager.set_capabilities().
"""

from __future__ import annotations

from ..capabilities import TrainingParadigm, GradientAvailability
from .step_trainer import StepTrainer


class LocalLearningStepTrainer(StepTrainer):
    """Step-based training for local learning paradigms (predictive coding).

    Inherits the entire StepTrainer training loop. Overrides only the
    capability class attributes so that HookManager correctly filters
    hooks based on LOCAL_LEARNING paradigm requirements.
    """

    paradigm = TrainingParadigm.LOCAL_LEARNING
    gradient_availability = GradientAvailability.LOCAL_GRADIENTS
