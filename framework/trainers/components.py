"""Training component bundles.

TrainingComponents is the ABC that trainers interact with. Paradigm-specific
subclasses (BackpropComponents, etc.) implement the interface. The trainer
never needs to know what paradigm it's running — it calls the interface.

Strategies ARE paradigm-aware and access public fields on concrete subclasses
(e.g., BackpropComponents.model, BackpropComponents.optimizer).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class TrainingComponents(ABC):
    """Paradigm-typed component bundle.

    Trainers interact with this through abstract methods — they never need
    to know the specific paradigm. Strategies and hooks interact with
    concrete subclasses directly, since they ARE paradigm-specific.

    Direct fields (trainer needs these for dispatch):
        strategy: The StrategyRunner that executes each training step.
        data: The data source (DataLoader, list, DataPool, etc.).
    """

    strategy: Any
    data: Any

    # === Mode switching ===

    @abstractmethod
    def train_mode(self) -> None:
        """Switch all models to training mode."""
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        """Switch all models to evaluation mode."""
        ...

    # === Scheduler ===

    @abstractmethod
    def step_schedulers(self, **kwargs) -> None:
        """Advance all LR schedulers one step."""
        ...

    @abstractmethod
    def get_lr(self) -> float | dict[str, float]:
        """Current learning rate(s)."""
        ...

    # === State persistence ===

    @abstractmethod
    def state_dict(self) -> dict:
        """Collect state dicts from all stateful components."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore all stateful components from a state dict."""
        ...

    # === Model access ===

    @abstractmethod
    def get_primary_model(self) -> nn.Module:
        """Return the primary model for evaluation and final saving."""
        ...

    @abstractmethod
    def get_device(self) -> torch.device:
        """Device the primary model lives on."""
        ...

    @abstractmethod
    def parameter_count(self) -> int:
        """Total trainable parameter count across all models."""
        ...

    # === Gradient management ===

    @abstractmethod
    def clip_gradients(self) -> dict | None:
        """Clip gradients on all trainable models.

        Returns:
            Dict with clipping info (e.g. {'grad_norm': float}) when clipping
            was applied, or None when no clipping is configured.
        """
        ...

    @abstractmethod
    def build_gradient_state(self, **kwargs):
        """Construct the paradigm-specific gradient state object for hooks.

        E.g., BackpropGradientState for backprop, AdversarialGradientState for GANs.
        """
        ...

    # === Emergency checkpoint ===

    @abstractmethod
    def capture_for_emergency(self) -> dict:
        """CPU snapshot of all stateful components for emergency save."""
        ...

    @abstractmethod
    def restore_from_emergency(self, snapshot: dict) -> None:
        """Restore from an emergency snapshot."""
        ...

    # === Hook context factories ===

    @abstractmethod
    def build_model_state(self):
        """Return paradigm-specific model state object for hooks.

        E.g., BackpropModelState for backprop, AdversarialModelState for GANs.
        """
        ...

    @abstractmethod
    def build_intervention_context_kwargs(self, *, loader, config, device,
                                           pre_epoch_state=None,
                                           current_batch=None,
                                           profiler=None) -> dict:
        """Return kwargs for constructing the paradigm-specific intervention context."""
        ...


@dataclass
class BackpropComponents(TrainingComponents):
    """Single model, single optimizer, single scheduler.

    Covers: standard backprop, gradient-aligned strategies, curriculum learning.
    """

    model: nn.Module
    optimizer: Any              # torch.optim.Optimizer
    scheduler: Any              # LR scheduler
    criterion: nn.Module | None
    loss_fn: Any                # (model, batch) -> loss, or None
    strategy: Any               # StrategyRunner
    data: Any                   # DataLoader / list / DataPool
    get_shuffled_loader_fn: Any = None  # (loader, config) -> shuffled_loader, or None
    max_grad_norm: float | None = None
    grad_scaler: Any = None             # torch.amp.GradScaler | None
    auxiliary_models: dict = None        # {name: nn.Module} — frozen/helper models

    def __post_init__(self):
        if self.auxiliary_models is None:
            self.auxiliary_models = {}

    # --- Mode switching ---

    def train_mode(self) -> None:
        self.model.train()
        for aux in self.auxiliary_models.values():
            aux.eval()  # auxiliary models (e.g. teacher) stay in eval

    def eval_mode(self) -> None:
        self.model.eval()
        for aux in self.auxiliary_models.values():
            aux.eval()

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
            self.model.parameters(), self.max_grad_norm
        )
        return {'grad_norm': float(grad_norm)}

    def build_gradient_state(self, **kwargs):
        from ..contexts.gradient_state import BackpropGradientState
        return BackpropGradientState(**kwargs)

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
        # Deferred import to avoid circular dependency with contexts package
        from ..contexts.model_state import BackpropModelState
        return BackpropModelState(model=self.model, lr=self.get_lr())

    def build_intervention_context_kwargs(self, *, loader, config, device,
                                           pre_epoch_state=None,
                                           current_batch=None,
                                           profiler=None) -> dict:
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'criterion': self.criterion,
            'loader': loader,
            'config': config,
            'device': device,
            'pre_epoch_state': pre_epoch_state,
            'current_batch': current_batch,
            'profiler': profiler,
            'loss_fn': self.loss_fn,
            'get_shuffled_loader_fn': self.get_shuffled_loader_fn,
            'grad_scaler': self.grad_scaler,
        }
