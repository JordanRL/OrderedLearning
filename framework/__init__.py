"""Experiment framework for ordered/curriculum learning research.

Provides reusable components for defining and running training experiments:
- StrategyRunner: defines training algorithms (SimpleTrainStep, etc.)
- ExperimentRunner: orchestrates experiments (LMRunner, GrokkingRunner)
- DatasetGenerator / DatasetLoader: data pipeline abstractions

Also provides shared utilities extracted from experiment files:
- Data pools, model configs, gradient selectors, display helpers, CLI helpers

Hook system infrastructure:
- Hook base types, registry, and manager (framework.hooks)
- Training contexts (framework.contexts)
- Metric sinks (framework.sinks)
"""

# Core abstractions
from .config import BaseConfig
from .eval import EvalResult
from .data import DatasetGenerator
from .data import DatasetLoader, FixedPoolLoader
from .data import RolloutBuffer, RolloutBatch
from .data import TaskSampler, TaskBatch
from .strategies import StrategyRunner, SimpleTrainStep, StepResult
from .strategies import (
    GradientAlignedStep, FixedTargetStep, PhasedCurriculumStep,
)
from .strategies import DistillationTrainStep
from .strategies import ContrastiveTrainStep, MomentumContrastiveTrainStep
from .strategies import AdversarialTrainStep, WGANGPTrainStep
from .strategies import PPOTrainStep, A2CTrainStep
from .strategies import PredictiveCodingTrainStep
from .strategies import MAMLStep, ReptileStep
from .strategies import EvolutionStrategyStep, GeneticAlgorithmStep
from .experiment_runner import ExperimentRunner, LMRunner, GrokkingRunner
from .trainers import (
    Trainer, TrainingComponents, BackpropComponents,
    StepTrainer, EpochTrainer,
    AdversarialComponents, AdversarialStepTrainer,
    RLComponents, RolloutTrainer,
    PredictiveCodingComponents, LocalLearningStepTrainer,
    MetaLearningComponents, MetaLearningStepTrainer,
    EvolutionaryComponents, EvolutionaryStepTrainer,
)
from .capabilities import (
    TrainingParadigm, ModelCapability, GradientAvailability,
    HookNeeds, TrainingCapabilities, HookRequirements,
)
from .checkpoints import ResumeInfo, ReferenceWeights
from .registry import Registry, ExperimentRegistry

# Hook system infrastructure
from .hooks import (
    HookPoint, MetricInfo, StepSchedule,
    TrainingHook, InterventionHook, DebugInterventionHook,
    HookRegistry, HookManager,
)
from .contexts import (
    RunContext, BackpropModelState, BackpropGradientState,
    AdversarialModelState, AdversarialGradientState,
    RLModelState, RLGradientState,
    PredictiveCodingModelState, PredictiveCodingGradientState,
    MetaLearningModelState, MetaLearningGradientState,
    EvolutionaryModelState, EvolutionaryGradientState,
    EvalMetrics, BatchState, BackpropInterventionContext, CheckpointProfiler,
)
from .sinks import MetricSink, FilePathSink, ConsoleSink, CSVSink, JSONLSink, WandbSink

# Shared utilities
from .utils import (
    set_seeds, set_determinism, snapshot_params, get_gradient_vector,
    cosine_similarity, pad_sequences,
    format_human_readable, format_bytes,
    get_environment_info, check_environment_compatibility,
    create_accumulator, accumulate, finalize,
)
from .models import MODEL_CONFIGS, get_lr_scheduler
from .models import PCLayer, PCLayerConfig, PredictiveCodingNetwork
from .data import (
    DataPool, FixedDataPool, DATASET_CONFIGS,
    GradientSelector, AlignedSelector, AntiAlignedSelector, RandomSelector,
    compute_candidate_alignments_sequential,
)
from .curriculum import CurriculumManager
from .cli import (
    add_common_args, add_hook_args, add_eval_target_args,
    handle_hook_inspection, build_hook_manager,
    OLArgumentParser,
)

__all__ = [
    # Core abstractions
    'BaseConfig',
    'EvalResult',
    'DatasetGenerator',
    'DatasetLoader', 'FixedPoolLoader',
    'RolloutBuffer', 'RolloutBatch',
    'TaskSampler', 'TaskBatch',
    'StrategyRunner', 'SimpleTrainStep', 'StepResult',
    'GradientAlignedStep', 'FixedTargetStep', 'PhasedCurriculumStep',
    'DistillationTrainStep',
    'ContrastiveTrainStep', 'MomentumContrastiveTrainStep',
    'AdversarialTrainStep', 'WGANGPTrainStep',
    'PPOTrainStep', 'A2CTrainStep',
    'PredictiveCodingTrainStep',
    'MAMLStep', 'ReptileStep',
    'EvolutionStrategyStep', 'GeneticAlgorithmStep',
    'ExperimentRunner', 'LMRunner', 'GrokkingRunner',
    'Trainer', 'TrainingComponents', 'BackpropComponents',
    'StepTrainer', 'EpochTrainer',
    'AdversarialComponents', 'AdversarialStepTrainer',
    'RLComponents', 'RolloutTrainer',
    'PredictiveCodingComponents', 'LocalLearningStepTrainer',
    'MetaLearningComponents', 'MetaLearningStepTrainer',
    'EvolutionaryComponents', 'EvolutionaryStepTrainer',
    'TrainingParadigm', 'ModelCapability', 'GradientAvailability',
    'HookNeeds', 'TrainingCapabilities', 'HookRequirements',
    'ResumeInfo', 'ReferenceWeights',
    'Registry', 'ExperimentRegistry',
    # Hook system infrastructure
    'HookPoint', 'MetricInfo', 'StepSchedule',
    'TrainingHook', 'InterventionHook', 'DebugInterventionHook',
    'HookRegistry', 'HookManager',
    'RunContext', 'BackpropModelState', 'BackpropGradientState',
    'AdversarialModelState', 'AdversarialGradientState',
    'RLModelState', 'RLGradientState',
    'PredictiveCodingModelState', 'PredictiveCodingGradientState',
    'MetaLearningModelState', 'MetaLearningGradientState',
    'EvolutionaryModelState', 'EvolutionaryGradientState',
    'EvalMetrics', 'BatchState', 'BackpropInterventionContext', 'CheckpointProfiler',
    'MetricSink', 'FilePathSink', 'ConsoleSink', 'CSVSink', 'JSONLSink', 'WandbSink',
    # Utilities
    'set_seeds', 'set_determinism', 'snapshot_params', 'get_gradient_vector',
    'cosine_similarity', 'pad_sequences',
    'format_human_readable', 'format_bytes',
    'create_accumulator', 'accumulate', 'finalize',
    'MODEL_CONFIGS', 'get_lr_scheduler',
    'PCLayer', 'PCLayerConfig', 'PredictiveCodingNetwork',
    'DataPool', 'FixedDataPool', 'DATASET_CONFIGS',
    'GradientSelector', 'AlignedSelector', 'AntiAlignedSelector', 'RandomSelector',
    'compute_candidate_alignments_sequential',
    'CurriculumManager',
    'add_common_args', 'add_hook_args', 'add_eval_target_args',
    'handle_hook_inspection', 'build_hook_manager',
    'OLArgumentParser',
]
