"""Reinforcement learning strategies.

Policy optimization strategies that operate on minibatches from a
rollout buffer. These do NOT collect rollouts — that's the trainer's
responsibility. Each train_step processes one minibatch.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .strategy_runner import StrategyRunner, StepResult
from ..capabilities import TrainingParadigm


class PPOTrainStep(StrategyRunner):
    """Proximal Policy Optimization (clipped surrogate objective).

    Each step processes one minibatch from the rollout buffer:
    1. Forward actor and critic on batch observations
    2. Compute clipped surrogate loss (policy) + value loss + entropy bonus
    3. Backward and step

    Setup expects:
        - components: RLComponents
        - kwargs['clip_range']: PPO clip range epsilon (default 0.2)
        - kwargs['value_coeff']: value loss coefficient (default 0.5)
        - kwargs['entropy_coeff']: entropy bonus coefficient (default 0.01)
    """

    paradigm = TrainingParadigm.ROLLOUT

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.actor = components.actor
        self.critic = components.critic
        self.optimizer = components.optimizer
        self.device = device
        self._components = components

        self.clip_range = kwargs.get('clip_range', 0.2)
        self.value_coeff = kwargs.get('value_coeff', 0.5)
        self.entropy_coeff = kwargs.get('entropy_coeff', 0.01)

        # AMP
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        """Process one minibatch from the rollout buffer.

        Args:
            step: Global step number.
            batch: RolloutBatch from the rollout buffer's get_batches().
        """
        if batch is None:
            return StepResult(metrics={'loss': 0.0}, trained=False)

        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            # Actor: get new log probs and entropy
            action_dist = self.actor(observations)
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.critic(observations).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            # Combined loss
            loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            self._components.clip_gradients()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self._components.clip_gradients()
            self.optimizer.step()

        return StepResult(
            metrics={
                'loss': loss.detach(),
                'policy_loss': policy_loss.detach(),
                'value_loss': value_loss.detach(),
                'entropy': entropy.detach(),
                'approx_kl': ((ratio - 1) - ratio.log()).mean().detach(),
            },
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "PPOTrainStep"


class A2CTrainStep(StrategyRunner):
    """Advantage Actor-Critic (synchronous).

    Simpler than PPO — no clipping. Uses the advantage directly as
    the policy gradient weight.

    Setup expects:
        - components: RLComponents
        - kwargs['value_coeff']: value loss coefficient (default 0.5)
        - kwargs['entropy_coeff']: entropy bonus coefficient (default 0.01)
    """

    paradigm = TrainingParadigm.ROLLOUT

    def setup(self, *, components, config, device, **kwargs) -> None:
        self.actor = components.actor
        self.critic = components.critic
        self.optimizer = components.optimizer
        self.device = device
        self._components = components

        self.value_coeff = kwargs.get('value_coeff', 0.5)
        self.entropy_coeff = kwargs.get('entropy_coeff', 0.01)

        # AMP
        self._scaler = components.grad_scaler
        self._use_amp = self._scaler is not None
        self._autocast_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'

    def train_step(self, step: int, batch: Any = None) -> StepResult:
        if batch is None:
            return StepResult(metrics={'loss': 0.0}, trained=False)

        observations = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)

        self.optimizer.zero_grad()

        with torch.autocast(device_type=self._autocast_device, enabled=self._use_amp):
            # Actor
            action_dist = self.actor(observations)
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()
            policy_loss = -(log_probs * advantages.detach()).mean()

            # Critic
            values = self.critic(observations).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy

        if self._scaler:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            self._components.clip_gradients()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            self._components.clip_gradients()
            self.optimizer.step()

        return StepResult(
            metrics={
                'loss': loss.detach(),
                'policy_loss': policy_loss.detach(),
                'value_loss': value_loss.detach(),
                'entropy': entropy.detach(),
            },
            batch_data=batch,
        )

    @property
    def name(self) -> str:
        return "A2CTrainStep"
