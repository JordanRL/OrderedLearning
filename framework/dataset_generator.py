"""Dataset generation abstractions.

DatasetGenerator provides a dataset. It is agnostic to how the data will
be ordered or consumed — that's the DatasetLoader's job.
"""

from abc import ABC, abstractmethod
from typing import Any


class DatasetGenerator(ABC):
    """Provides a dataset. Agnostic to how it will be ordered or consumed."""

    @abstractmethod
    def generate(self, config, **kwargs) -> Any:
        """Return a dataset object (torch Dataset, HF Dataset, list, etc.).

        Args:
            config: Experiment configuration (BaseConfig subclass).
            **kwargs: Additional keyword arguments for experiment-specific generators.

        Returns:
            A dataset object appropriate for the experiment.
        """
        ...
