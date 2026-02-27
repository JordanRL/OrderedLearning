"""Evolutionary step-based trainer.

Thin StepTrainer subclass that declares EVOLUTIONARY paradigm and
GradientAvailability.NONE. The training loop is identical to StepTrainer
— one generation maps to one step. The only difference is capability
declarations, which affect hook filtering via HookManager.set_capabilities().
"""

from __future__ import annotations

from ..capabilities import TrainingParadigm, GradientAvailability
from .step_trainer import StepTrainer


class EvolutionaryStepTrainer(StepTrainer):
    """Step-based training for evolutionary paradigms (ES, GA).

    Inherits the entire StepTrainer training loop. Overrides only the
    capability class attributes so that HookManager correctly filters
    hooks based on EVOLUTIONARY paradigm and NONE gradient requirements.
    """

    paradigm = TrainingParadigm.EVOLUTIONARY
    gradient_availability = GradientAvailability.NONE
