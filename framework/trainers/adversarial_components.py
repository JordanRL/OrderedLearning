"""Adversarial (GAN) training components.

Dual model/optimizer paradigm for generator-discriminator training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .components import TrainingComponents


@dataclass
class AdversarialComponents(TrainingComponents):
    """Generator + discriminator with separate optimizers and schedulers.

    Covers: standard GANs, Wasserstein GANs, conditional GANs, etc.
    """

    generator: nn.Module
    discriminator: nn.Module
    g_optimizer: Any           # torch.optim.Optimizer
    d_optimizer: Any           # torch.optim.Optimizer
    g_scheduler: Any = None    # LR scheduler for generator
    d_scheduler: Any = None    # LR scheduler for discriminator
    strategy: Any = None       # StrategyRunner
    data: Any = None           # DataLoader / list
    g_loss_fn: Any = None      # (G, D, batch) -> loss, or None
    d_loss_fn: Any = None      # (G, D, batch) -> loss, or None
    max_grad_norm: float | None = None
    grad_scaler: Any = None    # torch.amp.GradScaler | None
    d_steps_per_g_step: int = 1

    # --- Mode switching ---

    def train_mode(self) -> None:
        self.generator.train()
        self.discriminator.train()

    def eval_mode(self) -> None:
        self.generator.eval()
        self.discriminator.eval()

    # --- Scheduler ---

    def step_schedulers(self, **kwargs) -> None:
        if self.g_scheduler is not None:
            self.g_scheduler.step(**kwargs)
        if self.d_scheduler is not None:
            self.d_scheduler.step(**kwargs)

    def get_lr(self) -> dict[str, float]:
        g_lr = (self.g_scheduler.get_last_lr()[0]
                if self.g_scheduler and hasattr(self.g_scheduler, 'get_last_lr')
                else self.g_optimizer.param_groups[0]['lr'])
        d_lr = (self.d_scheduler.get_last_lr()[0]
                if self.d_scheduler and hasattr(self.d_scheduler, 'get_last_lr')
                else self.d_optimizer.param_groups[0]['lr'])
        return {'generator': g_lr, 'discriminator': d_lr}

    # --- State persistence ---

    def state_dict(self) -> dict:
        d = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict() if self.g_scheduler else None,
            'd_scheduler': self.d_scheduler.state_dict() if self.d_scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])
        self.g_optimizer.load_state_dict(state['g_optimizer'])
        self.d_optimizer.load_state_dict(state['d_optimizer'])
        if self.g_scheduler and state.get('g_scheduler'):
            self.g_scheduler.load_state_dict(state['g_scheduler'])
        if self.d_scheduler and state.get('d_scheduler'):
            self.d_scheduler.load_state_dict(state['d_scheduler'])
        if self.grad_scaler is not None and state.get('grad_scaler'):
            self.grad_scaler.load_state_dict(state['grad_scaler'])

    # --- Model access ---

    def get_primary_model(self) -> nn.Module:
        return self.generator

    def get_device(self) -> torch.device:
        return next(self.generator.parameters()).device

    def parameter_count(self) -> int:
        g_count = sum(p.numel() for p in self.generator.parameters())
        d_count = sum(p.numel() for p in self.discriminator.parameters())
        return g_count + d_count

    # --- Gradient management ---

    def clip_gradients(self) -> dict | None:
        if self.max_grad_norm is None:
            return None
        g_norm = torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(), self.max_grad_norm
        )
        d_norm = torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(), self.max_grad_norm
        )
        return {'g_grad_norm': float(g_norm), 'd_grad_norm': float(d_norm)}

    def build_gradient_state(self, **kwargs):
        from ..contexts.adversarial_state import AdversarialGradientState
        return AdversarialGradientState(**kwargs)

    # --- Emergency checkpoint ---

    def capture_for_emergency(self) -> dict:
        d = {
            'generator': {k: v.cpu().clone() for k, v in self.generator.state_dict().items()},
            'discriminator': {k: v.cpu().clone() for k, v in self.discriminator.state_dict().items()},
            'g_optimizer': copy.deepcopy(self.g_optimizer.state_dict()),
            'd_optimizer': copy.deepcopy(self.d_optimizer.state_dict()),
            'g_scheduler': self.g_scheduler.state_dict() if self.g_scheduler else None,
            'd_scheduler': self.d_scheduler.state_dict() if self.d_scheduler else None,
        }
        if self.grad_scaler is not None:
            d['grad_scaler'] = self.grad_scaler.state_dict()
        return d

    def restore_from_emergency(self, snapshot: dict) -> None:
        device = self.get_device()
        self.generator.load_state_dict(
            {k: v.to(device) for k, v in snapshot['generator'].items()}
        )
        self.discriminator.load_state_dict(
            {k: v.to(device) for k, v in snapshot['discriminator'].items()}
        )
        self.g_optimizer.load_state_dict(snapshot['g_optimizer'])
        self.d_optimizer.load_state_dict(snapshot['d_optimizer'])
        if self.g_scheduler and snapshot.get('g_scheduler'):
            self.g_scheduler.load_state_dict(snapshot['g_scheduler'])
        if self.d_scheduler and snapshot.get('d_scheduler'):
            self.d_scheduler.load_state_dict(snapshot['d_scheduler'])
        if self.grad_scaler is not None and snapshot.get('grad_scaler'):
            self.grad_scaler.load_state_dict(snapshot['grad_scaler'])

    # --- Hook context factories ---

    def build_model_state(self):
        from ..contexts.adversarial_state import AdversarialModelState
        return AdversarialModelState(
            generator=self.generator,
            discriminator=self.discriminator,
            lr=self.get_lr(),
        )

    def build_intervention_context_kwargs(self, *, loader, config, device,
                                           pre_epoch_state=None,
                                           current_batch=None,
                                           profiler=None) -> dict:
        # Adversarial training uses the generator as the intervention target
        return {
            'model': self.generator,
            'optimizer': self.g_optimizer,
            'scheduler': self.g_scheduler,
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
