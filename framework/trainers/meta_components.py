"""Meta-learning training components.

Single meta-model, single meta-optimizer, single scheduler.
The task_loss_fn replaces the standard loss_fn — it takes
(model_or_callable, batch) -> loss, where model_or_callable
may be either the raw model (Reptile) or a _ParameterizedModule
wrapper (MAML) for functional forward passes.

Strategies cache per-task adaptation data after each meta-step.
build_model_state() and build_gradient_state() read these caches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .components import TrainingComponents


@dataclass
class MetaLearningComponents(TrainingComponents):
    """Meta-model with single meta-optimizer.

    Covers: MAML, FOMAML, Reptile, and similar inner/outer loop methods.

    The strategy caches meta-gradients, per-task inner gradients, and
    task losses after each meta-step. build_model_state() and
    build_gradient_state() read these cached values to provide hooks
    with meta-learning-specific information.
    """

    model: nn.Module = None
    optimizer: Any = None
    scheduler: Any = None
    task_loss_fn: Any = None   # (model_or_callable, batch) -> loss
    strategy: Any = None
    data: Any = None           # TaskSampler
    max_grad_norm: float | None = None
    grad_scaler: Any = None    # torch.amp.GradScaler | None

    # --- Mode switching ---

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    # --- Scheduler ---

    def step_schedulers(self, **kwargs) -> None:
        if self.scheduler is not None:
            self.scheduler.step(**kwargs)

    def get_lr(self) -> float:
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()[0]
        return self.optimizer.param_groups[0]['lr']

    # --- State persistence ---

    def state_dict(self) -> dict:
        d = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and state.get('scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])
        if self.grad_scaler is not None and state.get('grad_scaler'):
            self.grad_scaler.load_state_dict(state['grad_scaler'])

    # --- Model access ---

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def parameter_count(self) -> int:
        raw = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
        return sum(p.numel() for p in raw.parameters())

    # --- Gradient management ---

    def clip_gradients(self) -> dict | None:
        if self.max_grad_norm is None:
            return None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        )
        return {'grad_norm': float(grad_norm)}

    def build_gradient_state(self, **kwargs):
        """Build meta-learning gradient state from cached strategy data.

        Unknown kwargs (like target_grad from StepTrainer) are ignored.
        """
        from ..contexts.meta_state import MetaLearningGradientState

        strategy = self.strategy
        return MetaLearningGradientState(
            meta_gradients=getattr(strategy, '_cached_meta_gradients', None),
            inner_gradients=getattr(strategy, '_cached_inner_gradients', None),
            task_losses=getattr(strategy, '_cached_task_losses', None),
        )

    # --- Emergency checkpoint ---

    def capture_for_emergency(self) -> dict:
        d = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def restore_from_emergency(self, snapshot: dict) -> None:
        device = self.get_device()
        self.model.load_state_dict(
            {k: v.to(device) for k, v in snapshot['model'].items()}
        )
        self.optimizer.load_state_dict(snapshot['optimizer'])
        if self.scheduler and snapshot.get('scheduler'):
            self.scheduler.load_state_dict(snapshot['scheduler'])
        if self.grad_scaler is not None and snapshot.get('grad_scaler'):
            self.grad_scaler.load_state_dict(snapshot['grad_scaler'])

    # --- Hook context factories ---

    def build_model_state(self):
        from ..contexts.meta_state import MetaLearningModelState

        strategy = self.strategy
        return MetaLearningModelState(
            model=self.model,
            lr=self.get_lr(),
            inner_lr=getattr(strategy, 'inner_lr', None),
        )

    def build_intervention_context_kwargs(
        self, *, loader, config, device,
        pre_epoch_state=None, current_batch=None, profiler=None,
    ) -> dict:
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'criterion': None,
            'loader': loader,
            'config': config,
            'device': device,
            'pre_epoch_state': pre_epoch_state,
            'current_batch': current_batch,
            'profiler': profiler,
            'loss_fn': self.task_loss_fn,
            'grad_scaler': self.grad_scaler,
        }
