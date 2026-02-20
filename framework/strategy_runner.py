"""Training strategy abstractions.

StrategyRunner defines the training algorithm. Owns optimizer interaction
(zero_grad, backward, step) but does NOT touch the scheduler — that's the
loop's responsibility (per-step or per-epoch).

Two modes of operation:
1. Batch-fed (epoch loop): Loop iterates data, calls train_step(step, batch)
2. Self-feeding (step loop): Loop calls train_step(step), strategy manages
   its own data access from the data source stored during setup()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Returned by StrategyRunner.train_step().

    Carries everything the loop needs for hooks, display, and control flow.
    """
    loss: Any                                           # training loss: detached tensor or float
    trained: bool = True                               # False if strategy skipped (e.g. empty pool)
    should_stop: bool = False                          # early stopping signal
    metrics: dict[str, float] = field(default_factory=dict)   # strategy-specific metrics
    batch_data: Any = None                             # raw batch tensor for hook contexts
    target_grad: Any = None                            # per-param target gradient dict (GradientAlignedStep)


class StrategyRunner(ABC):
    """Defines the training algorithm.

    Owns optimizer interaction (zero_grad, backward, step). Does NOT touch
    the scheduler — that belongs to the loop.
    """

    def setup(self, *, model, optimizer, config, device, **kwargs) -> None:
        """Called once before training. Store references needed for train_step.

        Common kwargs by strategy type:
            SimpleTrainStep:       criterion (epoch-based), data (step-based)
            GradientAlignedStep:   data, tokenizer, selector, target_config
        """
        pass

    @abstractmethod
    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Execute one training step.

        Args:
            step:  Global step number (1-indexed).
            batch: Pre-fetched batch, provided by the epoch loop.
                   None for step-loop strategies that manage own data.

        Returns:
            StepResult with loss, metrics, and control flags.
        """
        ...

    def post_step(self, step: int, result: StepResult) -> dict | None:
        """Strategy-specific post-step work (e.g. phase transition check).

        Called by the loop after hooks fire. Return dict of info for
        display/sinks, or None if nothing happened.
        """
        return None

    def teardown(self) -> None:
        """Called after training ends. Clean up state."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name for display."""
        return self.__class__.__name__


class SimpleTrainStep(StrategyRunner):
    """Standard forward-backward-step on the provided batch.

    Works for both epoch-based (mod arithmetic) and step-based (presorted)
    experiments. In step-loop mode, manages its own data iterator.
    """

    def setup(self, *, model, optimizer, config, device, **kwargs) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = kwargs.get('criterion')       # explicit loss fn (epoch-based)
        self.data = kwargs.get('data')                 # DataLoader (step-based only)
        self._data_iter = None

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        # If no batch provided (step loop), get from data iterator
        if batch is None:
            if self.data is None:
                raise ValueError("SimpleTrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(loss=0.0, trained=False, should_stop=True)

        self.optimizer.zero_grad()

        # Handle both dict-style (HuggingFace) and tensor-style (mod arithmetic) batches
        if isinstance(batch, dict):
            # HuggingFace-style: dict with input_ids, labels, attention_mask
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            loss = self.model(input_ids, labels=labels, attention_mask=attention_mask).loss
        else:
            # Tensor-style: batch[:, :2] = inputs, batch[:, 2] = targets
            inputs = batch[:, :2].to(self.device)
            targets = batch[:, 2].to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        loss.backward()
        self.optimizer.step()

        return StepResult(loss=loss.detach(), batch_data=batch)

    @property
    def name(self) -> str:
        return "SimpleTrainStep"
