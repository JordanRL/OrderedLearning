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

import torch

from ..capabilities import TrainingParadigm


@dataclass
class StepResult:
    """Returned by StrategyRunner.train_step().

    Carries everything the loop needs for hooks, display, and control flow.
    The primary carrier for training metrics is the ``metrics`` dict.
    The ``loss`` property provides convenience access to ``metrics['loss']``.

    Strategies populate metrics like:
        StepResult(metrics={'loss': loss.detach()}, batch_data=batch)
    """
    metrics: dict[str, Any] = field(default_factory=dict)
    trained: bool = True                               # False if strategy skipped (e.g. empty pool)
    should_stop: bool = False                          # early stopping signal
    batch_data: Any = None                             # raw batch tensor for hook contexts
    target_grad: Any = None                            # per-param target gradient dict (GradientAlignedStep)

    @property
    def loss(self) -> Any:
        """Convenience access to metrics['loss']. Returns None if not set."""
        return self.metrics.get('loss')


class StrategyRunner(ABC):
    """Defines the training algorithm.

    Owns optimizer interaction (zero_grad, backward, step). Does NOT touch
    the scheduler — that belongs to the loop.
    """

    paradigm: TrainingParadigm = TrainingParadigm.BACKPROP

    def setup(self, *, components, config, device, **kwargs) -> None:
        """Called once before training. Store references needed for train_step.

        Args:
            components: TrainingComponents bundle. Backprop strategies access
                BackpropComponents fields directly (model, optimizer, etc.).
            config: Experiment configuration.
            device: Training device.
            **kwargs: Strategy-specific args from runner.get_strategy_kwargs().
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

    Works for both epoch-based and step-based experiments. In step-loop
    mode, manages its own data iterator.

    Requires a loss_fn (provided by the experiment runner) that handles
    batch format: ``loss_fn(model, batch) -> loss_scalar``.
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self.loss_fn = components.loss_fn
        self.data = components.data
        self._components = components
        self._data_iter = None
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        # AMP setup
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'
        if self.loss_fn is None:
            raise ValueError(
                "SimpleTrainStep requires a loss_fn. "
                "Ensure the experiment's get_loss_fn() returns a callable."
            )

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
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            loss = self.loss_fn(self.model, batch)
        scaled_loss = loss / self._accumulation_steps

        if self._scaler:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._accum_count += 1
        if self._accum_count >= self._accumulation_steps:
            if self._scaler:
                self._scaler.unscale_(self.optimizer)
            self._components.clip_gradients()
            if self._scaler:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            self._accum_count = 0
            trained = True
        else:
            trained = False

        return StepResult(metrics={'loss': loss.detach()}, trained=trained, batch_data=batch)

    @property
    def name(self) -> str:
        return "SimpleTrainStep"
