"""Evolutionary training components.

Single template/center model, optional optimizer (for ES pseudo-gradient
updates), optional scheduler. The fitness_fn replaces the standard loss_fn
— it takes (model, data) -> float, evaluating a single individual.

Strategies cache per-generation fitness and pseudo-gradient data.
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
class EvolutionaryComponents(TrainingComponents):
    """Template model with optional optimizer for ES pseudo-gradient updates.

    Covers: Evolution Strategies (OpenAI-style ES) and Genetic Algorithms.

    The strategy caches pseudo-gradients (ES only), per-individual fitness
    values, and best/mean fitness after each generation. build_model_state()
    and build_gradient_state() read these cached values.
    """

    model: nn.Module = None
    optimizer: Any = None       # optional — ES uses it for pseudo-gradient updates
    scheduler: Any = None       # optional — for lr/sigma scheduling
    fitness_fn: Any = None      # (model, data) -> float
    strategy: Any = None
    data: Any = None            # evaluation dataset

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
        if self.optimizer is not None:
            if self.scheduler is not None and hasattr(self.scheduler, 'get_last_lr'):
                return self.scheduler.get_last_lr()[0]
            return self.optimizer.param_groups[0]['lr']
        return 0.0

    # --- State persistence ---

    def state_dict(self) -> dict:
        d = {'model': self.model.state_dict()}
        if self.optimizer is not None:
            d['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            d['scheduler'] = self.scheduler.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state['model'])
        if self.optimizer is not None and state.get('optimizer'):
            self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and state.get('scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])

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
        # No gradients in evolutionary paradigms
        return None

    def build_gradient_state(self, **kwargs):
        """Build evolutionary gradient state from cached strategy data.

        Unknown kwargs (like target_grad from StepTrainer) are ignored.
        """
        from ..contexts.evolutionary_state import EvolutionaryGradientState

        strategy = self.strategy
        return EvolutionaryGradientState(
            pseudo_gradient=getattr(strategy, '_cached_pseudo_gradient', None),
            fitness_values=getattr(strategy, '_cached_fitness_values', None),
        )

    # --- Emergency checkpoint ---

    def capture_for_emergency(self) -> dict:
        d = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
        }
        if self.optimizer is not None:
            d['optimizer'] = copy.deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            d['scheduler'] = self.scheduler.state_dict()
        return d

    def restore_from_emergency(self, snapshot: dict) -> None:
        device = self.get_device()
        self.model.load_state_dict(
            {k: v.to(device) for k, v in snapshot['model'].items()}
        )
        if self.optimizer is not None and snapshot.get('optimizer'):
            self.optimizer.load_state_dict(snapshot['optimizer'])
        if self.scheduler is not None and snapshot.get('scheduler'):
            self.scheduler.load_state_dict(snapshot['scheduler'])

    # --- Hook context factories ---

    def build_model_state(self):
        from ..contexts.evolutionary_state import EvolutionaryModelState

        strategy = self.strategy
        return EvolutionaryModelState(
            model=self.model,
            lr=self.get_lr() if self.optimizer is not None else None,
            best_fitness=getattr(strategy, '_cached_best_fitness', None),
            mean_fitness=getattr(strategy, '_cached_mean_fitness', None),
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
