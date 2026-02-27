"""Training loop implementations.

Provides the Trainer ABC, TrainingComponents hierarchy, and concrete
loop implementations (StepTrainer, EpochTrainer, AdversarialStepTrainer,
RolloutTrainer). Experiments never subclass Trainer — they implement
building blocks via ExperimentRunner.
"""

from .base import Trainer, LoopState, TrainResult
from .components import TrainingComponents, BackpropComponents
from .step_trainer import StepTrainer
from .epoch_trainer import EpochTrainer
from .adversarial_components import AdversarialComponents
from .adversarial_trainer import AdversarialStepTrainer
from .rl_components import RLComponents
from .rollout_trainer import RolloutTrainer
from .pc_components import PredictiveCodingComponents
from .local_learning_trainer import LocalLearningStepTrainer
from .meta_components import MetaLearningComponents
from .meta_trainer import MetaLearningStepTrainer
from .evolutionary_components import EvolutionaryComponents
from .evolutionary_trainer import EvolutionaryStepTrainer

__all__ = [
    'Trainer', 'LoopState', 'TrainResult',
    'TrainingComponents', 'BackpropComponents',
    'StepTrainer', 'EpochTrainer',
    'AdversarialComponents', 'AdversarialStepTrainer',
    'RLComponents', 'RolloutTrainer',
    'PredictiveCodingComponents', 'LocalLearningStepTrainer',
    'MetaLearningComponents', 'MetaLearningStepTrainer',
    'EvolutionaryComponents', 'EvolutionaryStepTrainer',
]
