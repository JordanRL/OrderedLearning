"""Shared fixtures for trainer tests — MockRunner and wired trainer instances."""

import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from framework.config import BaseConfig
from framework.eval.eval_result import EvalResult
from framework.capabilities import ModelCapability
from framework.trainers.components import BackpropComponents
from framework.trainers.adversarial_components import AdversarialComponents
from framework.trainers.rl_components import RLComponents
from framework.strategies.strategy_runner import SimpleTrainStep
from framework.strategies.adversarial import AdversarialTrainStep
from framework.strategies.rl import PPOTrainStep
from framework.data.rollout_buffer import RolloutBuffer


# ---- Config with small periodicities ----

@dataclass
class TrainerTestConfig(BaseConfig):
    """Config with small periodicities for fast tests."""
    steps: int = 5
    epochs: int = 3
    rollout_length: int = 4
    update_epochs: int = 2
    batch_size: int = 2


# ---- Helper models for RL testing ----

class TinyActor(nn.Module):
    """Minimal actor that returns a Categorical distribution."""

    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        logits = self.linear(x)
        return torch.distributions.Categorical(logits=logits)


class MockEnv:
    """Trivial environment satisfying reset/step protocol for RL testing."""

    def __init__(self, obs_dim=4):
        self.obs_dim = obs_dim

    def reset(self):
        return torch.zeros(self.obs_dim)

    def step(self, action):
        obs = torch.randn(self.obs_dim)
        reward = 1.0
        done = False
        return obs, reward, done, {}


# ---- MockRunner ----

class MockRunner:
    """Lightweight runner satisfying the ExperimentRunner interface for tests.

    Not a MagicMock — all methods are explicit so test failures are readable.
    Uses real BackpropComponents + SimpleTrainStep for genuine training.
    Supports component_type='backprop' (default), 'adversarial', or 'rl'.
    """

    def __init__(self, *, tmp_path, strategies=None, stop_at=None,
                 eval_every=2, snapshot_every=2, checkpoint_every=100,
                 steps=5, epochs=3, record_trajectory=False,
                 save_checkpoints=False, component_type='backprop'):
        self.config = TrainerTestConfig(
            eval_every=eval_every,
            snapshot_every=snapshot_every,
            checkpoint_every=checkpoint_every,
            steps=steps,
            epochs=epochs,
            record_trajectory=record_trajectory,
            save_checkpoints=save_checkpoints,
        )
        self.device = torch.device('cpu')
        self.model_capabilities = ModelCapability.PARAMETERS
        self.progress_metric = None
        self._strategies = strategies or ['test_strategy']
        self._stop_at = stop_at
        self._tmp_path = tmp_path
        self._evaluate_calls = []
        self._init_eval = None
        self._current_train_loader = None
        self._component_type = component_type

    def _build_fresh_components(self):
        """Build a fresh model/optimizer/strategy for each strategy run."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.LeakyReLU(), nn.Linear(4, 2))
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

        def loss_fn(m, batch):
            return m(batch).sum()

        batch = torch.randn(2, 4)
        strategy = SimpleTrainStep()

        # Enough batches for step trainer (self-feeding) and epoch trainer
        max_needed = max(self.config.steps, self.config.epochs * 2) + 5
        data = [batch.clone() for _ in range(max_needed)]

        return BackpropComponents(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=None,
            loss_fn=loss_fn,
            strategy=strategy,
            data=data,
        )

    def _build_adversarial_components(self):
        """Build adversarial (GAN) components for testing."""
        torch.manual_seed(42)
        generator = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        discriminator = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
        g_optimizer = SGD(generator.parameters(), lr=0.1)
        d_optimizer = SGD(discriminator.parameters(), lr=0.1)
        g_scheduler = StepLR(g_optimizer, step_size=1, gamma=0.9)
        d_scheduler = StepLR(d_optimizer, step_size=1, gamma=0.9)

        def g_loss_fn(G, D, batch):
            fake = G(batch)
            return -D(fake).mean()

        def d_loss_fn(G, D, batch):
            with torch.no_grad():
                fake = G(batch)
            return D(fake).mean() - D(batch).mean()

        batch = torch.randn(2, 4)
        max_needed = self.config.steps + 5
        data = [batch.clone() for _ in range(max_needed)]

        return AdversarialComponents(
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            g_scheduler=g_scheduler,
            d_scheduler=d_scheduler,
            strategy=AdversarialTrainStep(),
            data=data,
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

    def _build_rl_components(self):
        """Build RL (actor-critic) components for testing."""
        torch.manual_seed(42)
        actor = TinyActor(obs_dim=4, act_dim=2)
        critic = nn.Linear(4, 1)
        params = list(actor.parameters()) + list(critic.parameters())
        optimizer = SGD(params, lr=0.01)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
        env = MockEnv(obs_dim=4)
        buffer = RolloutBuffer(buffer_size=64)

        return RLComponents(
            actor=actor,
            critic=critic,
            optimizer=optimizer,
            scheduler=scheduler,
            strategy=PPOTrainStep(),
            data=env,
            rollout_buffer=buffer,
        )

    # --- Strategy/component lifecycle ---

    def get_strategies(self):
        return list(self._strategies)

    def build_components(self, strategy_name, total=None):
        if self._component_type == 'adversarial':
            return self._build_adversarial_components()
        elif self._component_type == 'rl':
            return self._build_rl_components()
        return self._build_fresh_components()

    def get_strategy_kwargs(self, strategy_name, components):
        return {}

    def get_total_steps(self):
        return self.config.steps

    def get_total_epochs(self):
        return self.config.epochs

    def get_epoch_loader(self, data, epoch):
        # Return exactly 2 batches per epoch for predictable test behavior
        return data[:2]

    # --- Evaluation ---

    def evaluate(self, model, step_or_epoch=None):
        self._evaluate_calls.append(step_or_epoch)
        return EvalResult(metrics={'loss': 0.1, 'acc': 0.5})

    def should_stop(self, counter, eval_result=None):
        if self._stop_at is not None:
            return counter >= self._stop_at
        return False

    # --- Hooks ---

    def wire_hooks(self, strategy_name, strategy, hook_manager):
        pass

    # --- Setup/teardown ---

    def setup_condition(self, strategy_name):
        pass

    def teardown_condition(self, strategy_name):
        pass

    # --- Display (all no-ops) ---

    def display_banner(self):
        pass

    def display_condition_start(self, strategy_name):
        pass

    def display_eval(self, step_or_epoch, eval_result, strategy_name):
        pass

    def display_post_step(self, step, post_info):
        pass

    def display_final(self, strategy_name, init_eval, final_eval):
        pass

    def display_comparison(self, all_results):
        pass

    # --- Output management ---

    def prepare_output_dir(self, strategy_name):
        d = os.path.join(str(self._tmp_path), strategy_name)
        os.makedirs(d, exist_ok=True)
        return d

    def save_config(self, experiment_dir, extra=None):
        return os.path.join(experiment_dir, 'experiment_config.json')

    def build_summary(self, strategy_name, init_eval, final_eval,
                      final_step_or_epoch, **kwargs):
        return {'strategy': strategy_name, 'final': final_step_or_epoch}

    def save_summary(self, experiment_dir, summary):
        return os.path.join(experiment_dir, 'summary.json')

    def save_trajectory(self, experiment_dir, trajectory):
        return None

    def save_final_model(self, experiment_dir, model, strategy_name):
        return os.path.join(experiment_dir, f'{strategy_name}_final.pt')

    def save_training_state(self):
        return None

    def load_training_state(self, state):
        pass


# ---- Fixtures ----

@pytest.fixture
def mock_runner(tmp_path):
    """MockRunner with default settings."""
    return MockRunner(tmp_path=tmp_path)


@pytest.fixture
def make_runner(tmp_path):
    """Factory fixture: make_runner(**kwargs) creates a MockRunner.

    Passes tmp_path automatically. Allows tests to customize
    strategies, steps, epochs, stop_at, etc.
    """
    def _make(**kwargs):
        return MockRunner(tmp_path=tmp_path, **kwargs)
    return _make
