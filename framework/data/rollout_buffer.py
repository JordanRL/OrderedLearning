"""Rollout buffer for RL training.

Stores transitions (obs, action, reward, done, log_prob, value),
computes GAE advantages, and provides batched iteration for policy updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class RolloutBuffer:
    """Stores rollout data and computes GAE advantages for policy optimization.

    Usage:
        buffer = RolloutBuffer(buffer_size=2048, gamma=0.99, gae_lambda=0.95)
        # During rollout collection:
        buffer.add(obs, action, reward, done, log_prob, value)
        # After rollout is complete:
        buffer.compute_returns(last_value)
        # During policy update:
        for batch in buffer.get_batches(batch_size=64):
            ...  # use batch.observations, batch.actions, etc.
        buffer.reset()
    """

    buffer_size: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_envs: int = 1

    # Internal storage (populated by add())
    observations: list = field(default_factory=list, repr=False)
    actions: list = field(default_factory=list, repr=False)
    rewards: list = field(default_factory=list, repr=False)
    dones: list = field(default_factory=list, repr=False)
    log_probs: list = field(default_factory=list, repr=False)
    values: list = field(default_factory=list, repr=False)

    # Computed by compute_returns()
    advantages: torch.Tensor | None = field(default=None, repr=False)
    returns: torch.Tensor | None = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def full(self) -> bool:
        return len(self.observations) >= self.buffer_size

    def add(self, obs, action, reward, done, log_prob, value):
        """Add a transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns(self, last_value: float | torch.Tensor = 0.0):
        """Compute GAE advantages and returns.

        Must be called after rollout collection is complete, before
        get_batches().

        Args:
            last_value: Value estimate for the state after the last transition.
                Used for bootstrapping when the rollout doesn't end at a terminal state.
        """
        n = len(self.observations)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        if isinstance(last_value, torch.Tensor):
            last_value = last_value.item()

        advantages = torch.zeros(n, dtype=torch.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + values

    def get_batches(self, batch_size: int):
        """Yield minibatches of rollout data for policy updates.

        Yields RolloutBatch namedtuples with tensors for each field.
        Data is randomly shuffled before batching.

        Args:
            batch_size: Number of transitions per batch.

        Yields:
            RolloutBatch with observations, actions, old_log_probs,
            advantages, returns.
        """
        if self.advantages is None:
            raise RuntimeError(
                "Must call compute_returns() before get_batches()."
            )

        n = len(self.observations)
        indices = torch.randperm(n)

        # Stack tensors
        observations = torch.stack([
            o if isinstance(o, torch.Tensor) else torch.tensor(o, dtype=torch.float32)
            for o in self.observations
        ])
        actions = torch.stack([
            a if isinstance(a, torch.Tensor) else torch.tensor(a)
            for a in self.actions
        ])
        old_log_probs = torch.stack([
            lp if isinstance(lp, torch.Tensor) else torch.tensor(lp, dtype=torch.float32)
            for lp in self.log_probs
        ])

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            yield RolloutBatch(
                observations=observations[batch_indices],
                actions=actions[batch_indices],
                old_log_probs=old_log_probs[batch_indices],
                advantages=self.advantages[batch_indices],
                returns=self.returns[batch_indices],
            )

    def reset(self):
        """Clear all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages = None
        self.returns = None


@dataclass
class RolloutBatch:
    """A minibatch from the rollout buffer."""
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
