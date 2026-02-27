"""Predictive coding training components.

Single model (PredictiveCodingNetwork), single optimizer with per-layer
parameter groups. The optimizer handles weight updates; the model handles
inference settling and local gradient computation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .components import TrainingComponents


@dataclass
class PredictiveCodingComponents(TrainingComponents):
    """PC model with single optimizer (per-layer parameter groups).

    The strategy caches settled activations and layer errors after each
    training step. build_model_state() and build_gradient_state() read
    these cached values to provide hooks with layer-level information.
    """

    model: nn.Module = None
    optimizer: Any = None
    scheduler: Any = None
    strategy: Any = None
    data: Any = None
    max_grad_norm: float | None = None

    # --- Mode switching ---

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    # --- Scheduler ---

    def step_schedulers(self, **kwargs) -> None:
        if self.scheduler is not None:
            self.scheduler.step(**kwargs)

    def get_lr(self) -> float | dict[str, float]:
        """Return LR. If per-layer groups, return dict keyed by group name."""
        if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
            lrs = self.scheduler.get_last_lr()
            groups = self.optimizer.param_groups
            if len(lrs) == 1:
                return lrs[0]
            return {
                g.get('name', f'layer_{i}'): lr
                for i, (g, lr) in enumerate(zip(groups, lrs))
            }
        groups = self.optimizer.param_groups
        if len(groups) == 1:
            return groups[0]['lr']
        return {
            g.get('name', f'layer_{i}'): g['lr']
            for i, g in enumerate(groups)
        }

    # --- State persistence ---

    def state_dict(self) -> dict:
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and state.get('scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])

    # --- Model access ---

    def get_primary_model(self) -> nn.Module:
        return self.model

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # --- Gradient management ---

    def clip_gradients(self) -> dict | None:
        if self.max_grad_norm is None:
            return None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm,
        )
        return {'grad_norm': float(grad_norm)}

    def build_gradient_state(self, **kwargs):
        """Build PC gradient state from cached strategy data.

        Unknown kwargs (like target_grad from StepTrainer) are ignored.
        """
        from ..contexts.pc_state import PredictiveCodingGradientState

        strategy = self.strategy
        return PredictiveCodingGradientState(
            layer_weight_grads=getattr(strategy, '_cached_weight_grads', None),
            layer_error_norms=getattr(strategy, '_cached_error_norms', None),
        )

    # --- Emergency checkpoint ---

    def capture_for_emergency(self) -> dict:
        return {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }

    def restore_from_emergency(self, snapshot: dict) -> None:
        device = self.get_device()
        self.model.load_state_dict(
            {k: v.to(device) for k, v in snapshot['model'].items()}
        )
        self.optimizer.load_state_dict(snapshot['optimizer'])
        if self.scheduler and snapshot.get('scheduler'):
            self.scheduler.load_state_dict(snapshot['scheduler'])

    # --- Hook context factories ---

    def build_model_state(self):
        from ..contexts.pc_state import PredictiveCodingModelState

        strategy = self.strategy
        return PredictiveCodingModelState(
            model=self.model,
            lr=self.get_lr(),
            settled_activations=getattr(strategy, '_cached_activations', None),
            layer_errors=getattr(strategy, '_cached_errors', None),
            free_energy=getattr(strategy, '_cached_free_energy', None),
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
            'loss_fn': None,
        }
