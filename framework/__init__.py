"""Experiment framework for ordered/curriculum learning research.

Provides reusable components for defining and running training experiments:
- StrategyRunner: defines training algorithms (SimpleTrainStep, etc.)
- ExperimentRunner: orchestrates experiments (LMRunner, GrokkingRunner)
- DatasetGenerator / DatasetLoader: data pipeline abstractions

Also provides shared utilities extracted from experiment files:
- Data pools, model configs, gradient selectors, display helpers, CLI helpers
"""

# Core abstractions
from .config import BaseConfig
from .eval_result import EvalResult
from .dataset_generator import DatasetGenerator
from .dataset_loader import DatasetLoader, FixedPoolLoader
from .strategy_runner import StrategyRunner, SimpleTrainStep, StepResult
from .gradient_aligned_step import (
    GradientAlignedStep, FixedTargetStep, PhasedCurriculumStep,
)
from .experiment_runner import ExperimentRunner, LMRunner, GrokkingRunner
from .loops import step_loop, epoch_loop
from .resume import ResumeInfo
from .registry import ExperimentRegistry

# Shared utilities
from .utils import (
    set_seeds, set_determinism, snapshot_params, get_gradient_vector,
    cosine_similarity, pad_sequences,
    format_human_readable, format_bytes,
    get_environment_info, check_environment_compatibility,
)
from .models import MODEL_CONFIGS, get_lr_scheduler
from .data import DataPool, FixedDataPool, DATASET_CONFIGS
from .selectors import (
    GradientSelector, AlignedSelector, AntiAlignedSelector, RandomSelector,
    compute_candidate_alignments_sequential,
)
from .curriculum import CurriculumManager
from .cli import (
    add_common_args, add_hook_args, add_eval_target_args,
    handle_hook_inspection, build_hook_manager,
)
from .cli_parser import OLArgumentParser

__all__ = [
    # Core abstractions
    'BaseConfig',
    'EvalResult',
    'DatasetGenerator',
    'DatasetLoader', 'FixedPoolLoader',
    'StrategyRunner', 'SimpleTrainStep', 'StepResult',
    'GradientAlignedStep', 'FixedTargetStep', 'PhasedCurriculumStep',
    'ExperimentRunner', 'LMRunner', 'GrokkingRunner',
    'step_loop', 'epoch_loop',
    'ResumeInfo',
    'ExperimentRegistry',
    # Utilities
    'set_seeds', 'set_determinism', 'snapshot_params', 'get_gradient_vector',
    'cosine_similarity', 'pad_sequences',
    'format_human_readable', 'format_bytes',
    'MODEL_CONFIGS', 'get_lr_scheduler',
    'DataPool', 'FixedDataPool', 'DATASET_CONFIGS',
    'GradientSelector', 'AlignedSelector', 'AntiAlignedSelector', 'RandomSelector',
    'compute_candidate_alignments_sequential',
    'CurriculumManager',
    'add_common_args', 'add_hook_args', 'add_eval_target_args',
    'handle_hook_inspection', 'build_hook_manager',
    'OLArgumentParser',
]
