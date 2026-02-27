"""Episodic task sampling for meta-learning.

TaskSampler is the ABC for experiments to implement. It provides
batches of tasks, each split into support (for inner adaptation)
and query (for meta-loss evaluation) sets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskBatch:
    """A single task with support and query splits.

    Fields are deliberately untyped beyond Any — the experiment
    defines what 'support' and 'query' contain (e.g., (inputs, targets)
    tuples, single tensors, dicts). The strategy's task_loss_fn knows
    how to consume them.
    """
    support: Any
    query: Any
    task_id: int | str | None = None


class TaskSampler(ABC):
    """ABC for episodic task sampling.

    Experiments subclass this to provide meta-learning tasks. The
    sampler is set as components.data, so the strategy can call
    sample() to get tasks each meta-step.

    Examples:
        - Few-shot classification: each task is a randomly chosen
          N-way K-shot episode from a task distribution.
        - Regression: each task is a random sinusoid with K support
          and K query points.
    """

    @abstractmethod
    def sample(self, n_tasks: int) -> list[TaskBatch]:
        """Sample n_tasks episodes, each with support/query splits.

        Args:
            n_tasks: Number of tasks to sample for this meta-step.

        Returns:
            List of TaskBatch instances.
        """
        ...
