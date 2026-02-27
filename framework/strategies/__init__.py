"""Training strategy abstractions and implementations.

Provides StrategyRunner (ABC), SimpleTrainStep (standard training),
gradient-aligned strategies, distillation, contrastive, adversarial,
RL, self-supervised, generative, preference, and multi-task strategies.
"""

from .strategy_runner import StrategyRunner, SimpleTrainStep, StepResult
from .gradient_aligned_step import (
    GradientAlignedStep, FixedTargetStep, PhasedCurriculumStep,
)
from .distillation import DistillationTrainStep
from .contrastive import ContrastiveTrainStep, MomentumContrastiveTrainStep
from .adversarial import AdversarialTrainStep, WGANGPTrainStep
from .rl import PPOTrainStep, A2CTrainStep
from .predictive_coding import PredictiveCodingTrainStep
from .meta_learning import MAMLStep, ReptileStep
from .evolutionary import EvolutionStrategyStep, GeneticAlgorithmStep
from .vae import VAETrainStep
from .diffusion import DDPMTrainStep
from .masked_modeling import MaskedModelingTrainStep
from .self_distillation import SelfDistillationTrainStep
from .dpo import DPOTrainStep
from .multi_task import MultiTaskTrainStep

__all__ = [
    'StrategyRunner', 'SimpleTrainStep', 'StepResult',
    'GradientAlignedStep', 'FixedTargetStep', 'PhasedCurriculumStep',
    'DistillationTrainStep',
    'ContrastiveTrainStep', 'MomentumContrastiveTrainStep',
    'AdversarialTrainStep', 'WGANGPTrainStep',
    'PPOTrainStep', 'A2CTrainStep',
    'PredictiveCodingTrainStep',
    'MAMLStep', 'ReptileStep',
    'EvolutionStrategyStep', 'GeneticAlgorithmStep',
    'VAETrainStep',
    'DDPMTrainStep',
    'MaskedModelingTrainStep',
    'SelfDistillationTrainStep',
    'DPOTrainStep',
    'MultiTaskTrainStep',
]
