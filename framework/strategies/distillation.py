"""Knowledge distillation strategy.

Teacher-student training: the teacher is frozen, the student learns from
a combination of hard loss (ground truth) and soft loss (teacher logits).

Uses existing BackpropComponents with auxiliary_models={'teacher': ...}.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult


class DistillationTrainStep(StrategyRunner):
    """Teacher-student distillation with combined hard + soft loss.

    Setup expects:
        - components.model: the student model
        - components.auxiliary_models['teacher'] or kwargs['teacher']: frozen teacher
        - kwargs['temperature']: distillation temperature (default 4.0)
        - kwargs['alpha']: weight for hard loss vs soft loss (default 0.5)
            loss = alpha * hard_loss + (1 - alpha) * T^2 * kl_loss
        - components.loss_fn: hard loss function (model, batch) -> loss
    """

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.model = components.model
        self.optimizer = components.optimizer
        self.device = device
        self._components = components
        self._data_iter = None

        # Teacher model
        self.teacher = kwargs.get('teacher') or components.auxiliary_models.get('teacher')
        if self.teacher is None:
            raise ValueError(
                "DistillationTrainStep requires a teacher model. "
                "Provide via kwargs['teacher'] or components.auxiliary_models['teacher']."
            )
        self.teacher.eval()

        # Distillation parameters
        self.temperature = kwargs.get('temperature', 4.0)
        self.alpha = kwargs.get('alpha', 0.5)

        # Hard loss function
        self.hard_loss_fn = components.loss_fn
        if self.hard_loss_fn is None:
            raise ValueError(
                "DistillationTrainStep requires a loss_fn for hard loss. "
                "Ensure the experiment's get_loss_fn() returns a callable."
            )

        # Soft loss function: researcher can override
        self.soft_loss_fn = kwargs.get('soft_loss_fn', self._default_soft_loss)

        self.data = components.data

        # Accumulation + AMP (follow SimpleTrainStep pattern)
        self._accumulation_steps = getattr(config, 'accumulation_steps', 1)
        self._accum_count = 0
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def _default_soft_loss(self, student_logits, teacher_logits):
        """Default KL divergence soft loss."""
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            if self.data is None:
                raise ValueError("DistillationTrainStep: no batch and no data source")
            if self._data_iter is None:
                self._data_iter = iter(self.data)
            try:
                batch = next(self._data_iter)
            except StopIteration:
                return StepResult(metrics={'loss': 0.0}, trained=False, should_stop=True)

        if self._accum_count == 0:
            self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_out = self.teacher(batch) if not hasattr(self.teacher, 'forward') \
                    else self.teacher(batch)
                teacher_logits = teacher_out.logits if hasattr(teacher_out, 'logits') else teacher_out

            # Student forward
            student_out = self.model(batch) if not hasattr(self.model, 'forward') \
                else self.model(batch)
            student_logits = student_out.logits if hasattr(student_out, 'logits') else student_out

            # Combined loss
            hard_loss = self.hard_loss_fn(self.model, batch)
            soft_loss = self.soft_loss_fn(student_logits, teacher_logits)
            loss = self.alpha * hard_loss + (1 - self.alpha) * (self.temperature ** 2) * soft_loss

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

        return StepResult(
            metrics={
                'loss': loss.detach(),
                'hard_loss': hard_loss.detach(),
                'soft_loss': soft_loss.detach(),
            },
            trained=trained,
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "DistillationTrainStep"
