"""Data pipeline abstractions: pools, generators, loaders, selectors, and buffers.

Provides DataPool/FixedDataPool for dataset management, DatasetGenerator
and DatasetLoader ABCs for experiment data pipelines, GradientSelector
hierarchy for gradient-aligned data selection, and RolloutBuffer for RL.
"""

from .pools import DataPool, FixedDataPool, DATASET_CONFIGS
from .dataset_generator import DatasetGenerator
from .dataset_loader import DatasetLoader, FixedPoolLoader
from .selectors import (
    GradientSelector, AlignedSelector, AntiAlignedSelector, RandomSelector,
    compute_candidate_alignments_sequential,
)
from .rollout_buffer import RolloutBuffer, RolloutBatch
from .task_sampler import TaskSampler, TaskBatch

__all__ = [
    'DataPool', 'FixedDataPool', 'DATASET_CONFIGS',
    'DatasetGenerator',
    'DatasetLoader', 'FixedPoolLoader',
    'GradientSelector', 'AlignedSelector', 'AntiAlignedSelector', 'RandomSelector',
    'compute_candidate_alignments_sequential',
    'RolloutBuffer', 'RolloutBatch',
    'TaskSampler', 'TaskBatch',
]
