"""Multi-task learning strategy with dynamic task weighting.

Supports three weighting modes:
- 'uncertainty': Learnable log-variance parameters (Kendall et al., 2018)
- 'uniform': Equal weight for all tasks
- 'fixed': User-specified static weights

Uses existing BackpropComponents. The runner provides task_loss_fns via kwargs.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .strategy_runner import StrategyRunner, StepResult


class MultiTaskTrainStep(StrategyRunner):
    """Multi-task training with dynamic loss weighting.

    Each step:
    1. Compute each task's loss via task_loss_fns
    2. Weight losses according to the selected scheme
    3. Sum weighted losses and backward
    4. Step optimizer

    Setup expects:
        - kwargs['task_loss_fns']: dict[str, callable] — {name: fn(model, batch) -> loss}.
            Must contain at least 2 tasks.
        - kwargs['weighting']: 'uncertainty' | 'uniform' | 'fixed' (default 'uncertainty')
        - kwargs['fixed_weights']: dict[str, float] | None — required when weighting='fixed'

    For 'uncertainty' weighting, the strategy creates learnable log-variance
    parameters (one per task) and adds them to the optimizer as a new param group.
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None
        self.data = components.data

        # Task loss functions (required)
        self.task_loss_fns = kwargs.get('task_loss_fns')
        if self.task_loss_fns is None:
            raise ValueError(
                "MultiTaskTrainStep requires task_loss_fns. "
                "Provide via kwargs['task_loss_fns'] as dict[str, callable]."
            )
        if len(self.task_loss_fns) < 2:
            raise ValueError(
                "MultiTaskTrainStep requires at least 2 tasks, "
                f"got {len(self.task_loss_fns)}."
            )

        self.task_names = sorted(self.task_loss_fns.keys())
        self.weighting = kwargs.get('weighting', 'uncertainty')

        # Weighting mode setup
        if self.weighting == 'uncertainty':
            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.zeros(1, device=device))
                for name in self.task_names
            })
            # Add learnable weights to the optimizer as a new param group
            self.optimizer.add_param_group({
                'params': list(self.log_vars.parameters()),
                'lr': self.optimizer.defaults.get('lr', 0.01),
            })
        elif self.weighting == 'fixed':
            self.fixed_weights = kwargs.get('fixed_weights')
            if self.fixed_weights is None:
                raise ValueError(
                    "MultiTaskTrainStep with weighting='fixed' requires fixed_weights. "
                    "Provide via kwargs['fixed_weights'] as dict[str, float]."
                )
        elif self.weighting != 'uniform':
            raise ValueError(
                f"MultiTaskTrainStep: unknown weighting '{self.weighting}'. "
                "Use 'uncertainty', 'uniform', or 'fixed'."
            )

        # Accumulation + AMP
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _compute_weighted_loss(self, task_losses):
        """Compute weighted total loss and effective weights.

        Returns:
            (total_loss, weights_dict) where weights_dict maps task name
            to its effective weight.
        """
        if self.weighting == 'uncertainty':
            total = torch.tensor(0.0, device=self.device)
            weights = {}
            for name in self.task_names:
                log_var = self.log_vars[name]
                precision = torch.exp(-log_var)
                total = total + 0.5 * precision * task_losses[name] + 0.5 * log_var
                weights[name] = precision.detach().item()
            return total, weights

        elif self.weighting == 'uniform':
            n = len(self.task_names)
            total = sum(task_losses[name] for name in self.task_names) / n
            weights = {name: 1.0 / n for name in self.task_names}
            return total, weights

        else:  # fixed
            total = torch.tensor(0.0, device=self.device)
            weights = {}
            for name in self.task_names:
                w = self.fixed_weights.get(name, 1.0)
                total = total + w * task_losses[name]
                weights[name] = w
            return total, weights

    def get_task_weights(self):
        """Return current effective weights per task.

        For 'uncertainty': exp(-log_var) values.
        For 'uniform': 1/N.
        For 'fixed': the fixed weights.
        """
        if self.weighting == 'uncertainty':
            return {
                name: math.exp(-self.log_vars[name].item())
                for name in self.task_names
            }
        elif self.weighting == 'uniform':
            n = len(self.task_names)
            return {name: 1.0 / n for name in self.task_names}
        else:
            return {name: self.fixed_weights.get(name, 1.0) for name in self.task_names}

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("MultiTaskTrainStep: no batch provided and no data source set")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            # Compute all task losses
            task_losses = {}
            for name in self.task_names:
                task_losses[name] = self.task_loss_fns[name](self.model, batch)

            # Weighted combination
            loss, weights = self._compute_weighted_loss(task_losses)

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

        # Build metrics
        metrics = {'loss': loss.detach()}
        for name in self.task_names:
            metrics[f'task_{name}_loss'] = task_losses[name].detach()
            metrics[f'task_{name}_weight'] = weights[name]

        return StepResult(metrics=metrics, trained=trained, batch_data=batch)

    @property
    def name(self) -> str:
        return "MultiTaskTrainStep"
