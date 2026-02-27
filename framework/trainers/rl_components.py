"""Reinforcement learning training components.

Actor-critic paradigm with rollout buffer and environment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .components import TrainingComponents


@dataclass
class RLComponents(TrainingComponents):
    """Actor-critic with shared or separate optimizers.

    Covers: PPO, A2C, and similar policy gradient methods.
    The environment protocol is minimal — experiments provide it.
    """

    actor: nn.Module
    critic: nn.Module
    optimizer: Any              # torch.optim.Optimizer (shared or actor-only)
    scheduler: Any = None       # LR scheduler
    strategy: Any = None        # StrategyRunner (PPOTrainStep, A2CTrainStep)
    data: Any = None            # Environment (reset/step protocol)
    rollout_buffer: Any = None  # RolloutBuffer
    max_grad_norm: float | None = None
    grad_scaler: Any = None     # torch.amp.GradScaler | None

    # --- Mode switching ---

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic.eval()

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
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler and state.get('scheduler'):
            self.scheduler.load_state_dict(state['scheduler'])
        if self.grad_scaler is not None and state.get('grad_scaler'):
            self.grad_scaler.load_state_dict(state['grad_scaler'])

    # --- Model access ---

    def get_primary_model(self) -> nn.Module:
        return self.actor

    def get_device(self) -> torch.device:
        return next(self.actor.parameters()).device

    def parameter_count(self) -> int:
        a_count = sum(p.numel() for p in self.actor.parameters())
        c_count = sum(p.numel() for p in self.critic.parameters())
        return a_count + c_count

    # --- Gradient management ---

    def clip_gradients(self) -> dict | None:
        if self.max_grad_norm is None:
            return None
        # Clip all parameters the optimizer manages
        all_params = list(self.actor.parameters()) + list(self.critic.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        return {'grad_norm': float(grad_norm)}

    def build_gradient_state(self, **kwargs):
        from ..contexts.rl_state import RLGradientState
        return RLGradientState(**kwargs)

    # --- Emergency checkpoint ---

    def capture_for_emergency(self) -> dict:
        d = {
            'actor': {k: v.cpu().clone() for k, v in self.actor.state_dict().items()},
            'critic': {k: v.cpu().clone() for k, v in self.critic.state_dict().items()},
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def restore_from_emergency(self, snapshot: dict) -> None:
        device = self.get_device()
        self.actor.load_state_dict(
            {k: v.to(device) for k, v in snapshot['actor'].items()}
        )
        self.critic.load_state_dict(
            {k: v.to(device) for k, v in snapshot['critic'].items()}
        )
        self.optimizer.load_state_dict(snapshot['optimizer'])
        if self.scheduler and snapshot.get('scheduler'):
            self.scheduler.load_state_dict(snapshot['scheduler'])
        if self.grad_scaler is not None and snapshot.get('grad_scaler'):
            self.grad_scaler.load_state_dict(snapshot['grad_scaler'])

    # --- Hook context factories ---

    def build_model_state(self):
        from ..contexts.rl_state import RLModelState
        return RLModelState(
            actor=self.actor,
            critic=self.critic,
            lr=self.get_lr(),
        )

    def build_intervention_context_kwargs(self, *, loader, config, device,
                                           pre_epoch_state=None,
                                           current_batch=None,
                                           profiler=None) -> dict:
        # RL uses the actor as the intervention target
        return {
            'model': self.actor,
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
            'grad_scaler': self.grad_scaler,
        }
