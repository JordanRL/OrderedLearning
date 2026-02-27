"""Dataset loading and ordering abstractions.

DatasetLoader provides ordered access to examples from a dataset.
Could produce a DataLoader (for batch iteration), a DataPool (for sampling),
or any iterable. The access pattern is defined by the loader.
"""

from abc import ABC, abstractmethod
from typing import Any


class DatasetLoader(ABC):
    """Provides ordered access to examples from a dataset.

    Could produce a DataLoader (for batch iteration), a DataPool (for sampling),
    or any iterable. The access pattern is defined by the loader.
    """

    @abstractmethod
    def load(self, dataset: Any, config, **kwargs) -> Any:
        """Return an ordered/accessible data source from the given dataset.

        Args:
            dataset: Dataset from DatasetGenerator.generate().
            config: Experiment configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A data source (DataLoader, DataPool, iterable, etc.)
        """
        ...


class FixedPoolLoader(DatasetLoader):
    """Wraps a DataPool into a FixedDataPool for deterministic sampling.

    Used by gradient-aligned experiments (guided_llm, phased_curriculum)
    where the strategy selects candidates from the pool each step.
    """

    def __init__(self, console=None):
        self.console = console

    def load(self, dataset, config, **kwargs):
        """Wrap DataPool in a FixedDataPool with step-seeded sampling.

        Args:
            dataset: A DataPool from a DatasetGenerator.
            config: Experiment config (needs steps, batch_size, seed).

        Returns:
            FixedDataPool with deterministic access.
        """
        from .pools import FixedDataPool
        total_examples = config.steps * config.batch_size
        return FixedDataPool(
            source_pool=dataset,
            batch_size=config.batch_size,
            total_examples_limit=total_examples,
            seed=config.seed,
            console=self.console,
        )
