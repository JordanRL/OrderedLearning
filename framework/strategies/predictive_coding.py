"""Predictive coding training strategy.

Implements the PC training loop: clamp observations, settle via iterative
inference, compute local weight gradients, optimizer step.
"""

from __future__ import annotations

from typing import Any

import torch

from .strategy_runner import StrategyRunner, StepResult
from ..capabilities import TrainingParadigm


class PredictiveCodingTrainStep(StrategyRunner):
    """Predictive coding training with iterative inference settling.

    Each training step:
    1. Transform batch into clamped observations via clamp_fn
    2. Initialize activations from clamped input
    3. Iterative inference settling (T steps under torch.no_grad)
    4. Compute local weight gradients (Hebbian rule, no autograd)
    5. Clip gradients, optimizer step
    6. Cache settled state for hooks

    Setup expects:
        - components: PredictiveCodingComponents
        - kwargs['clamp_fn']: callable(batch, device) -> dict[int, Tensor]
            Transforms a data batch into clamped observations keyed by
            layer index. Example: {0: input_tensor, L: label_tensor}.
        - kwargs['inference_steps']: settling iterations (default 20)
        - kwargs['inference_lr']: inference step size (default 0.1)
    """

    paradigm = TrainingParadigm.LOCAL_LEARNING

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self.data = components.data
        self._data_iter = None

        # Clamp function: experiment-provided, required
        self.clamp_fn = kwargs.get('clamp_fn')
        if self.clamp_fn is None:
            raise ValueError(
                "PredictiveCodingTrainStep requires a clamp_fn. "
                "Provide via runner.get_strategy_kwargs() returning "
                "{'clamp_fn': callable(batch, device) -> dict[int, Tensor]}."
            )

        # Inference hyperparameters
        self.inference_steps = kwargs.get('inference_steps', 20)
        self.inference_lr = kwargs.get('inference_lr', 0.1)

        # Strategy cache (read by PredictiveCodingComponents for hooks)
        self._cached_activations: list[torch.Tensor] | None = None
        self._cached_errors: list[torch.Tensor] | None = None
        self._cached_free_energy: float | None = None
        self._cached_weight_grads: list[torch.Tensor | None] | None = None
        self._cached_error_norms: list[float] | None = None

    def _get_batch(self) -> Any:
        """Get next batch from data iterator, resetting on exhaustion."""
        if self._data_iter is None:
            self._data_iter = iter(self.data)
        try:
            return next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.data)
            return next(self._data_iter)

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError(
                    "PredictiveCodingTrainStep: no batch provided and no data source"
                )
            batch = self._get_batch()

        # 1. Transform batch into clamped observations
        clamped = self.clamp_fn(batch, self.device)

        # 2. Initialize activations from the lowest clamped layer
        init_layer = min(clamped.keys())
        activations = self.model.initialize_activations(clamped[init_layer])

        # 3. Apply all clamped values
        for layer_idx, tensor in clamped.items():
            activations[layer_idx] = tensor.clone()

        # 4. Iterative inference settling
        with torch.no_grad():
            for _ in range(self.inference_steps):
                self.model.inference_step(activations, clamped, self.inference_lr)

        # 5. Compute free energy before weight update (for reporting)
        free_energy = self.model.free_energy(activations)

        # 6. Zero gradients and compute local weight gradients
        self.optimizer.zero_grad()
        self.model.compute_weight_gradients(activations)

        # 7. Clip gradients
        self._components.clip_gradients()

        # 8. Optimizer step
        self.optimizer.step()

        # 9. Cache state for hooks (read by PredictiveCodingComponents)
        errors = self.model.compute_layer_errors(activations)
        self._cached_activations = [a.detach() for a in activations]
        self._cached_errors = [e.detach() for e in errors]
        self._cached_free_energy = float(free_energy.detach())
        self._cached_weight_grads = [
            layer.generative.weight.grad.detach().clone()
            if layer.generative.weight.grad is not None else None
            for layer in self.model.layers
        ]
        self._cached_error_norms = [
            float(e.detach().norm()) for e in errors
        ]

        # 10. Build metrics
        metrics: dict[str, Any] = {
            'loss': free_energy.detach(),
            'free_energy': free_energy.detach(),
        }
        for i, norm in enumerate(self._cached_error_norms):
            metrics[f'layer_{i}_error_norm'] = norm

        return StepResult(metrics=metrics, batch_data=batch)

    @property
    def name(self) -> str:
        return "PredictiveCodingTrainStep"
