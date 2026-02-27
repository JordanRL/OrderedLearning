"""Shared fixtures for OrderedLearning unit tests."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from framework.config import BaseConfig
from framework.trainers.components import BackpropComponents
from framework.trainers.evolutionary_components import EvolutionaryComponents
from framework.trainers.adversarial_components import AdversarialComponents
from framework.trainers.rl_components import RLComponents
from framework.trainers.meta_components import MetaLearningComponents


# ---- Console singleton: force NULL mode before any test touches it ----

@pytest.fixture(autouse=True, scope="session")
def _silence_console():
    """Initialize OLConsole in NULL mode to suppress all output during tests.

    Must run before any code that calls OLConsole(). Session-scoped so
    the singleton is set once and stays NULL for the entire test run.

    To test OLConsole directly in the future, individual tests can
    re-initialize via OLConsole(ConsoleConfig(mode=...)) — the singleton
    re-initializes when passed a different config.
    """
    from console.config import ConsoleConfig, ConsoleMode
    from console.olconsole import OLConsole
    OLConsole(ConsoleConfig(mode=ConsoleMode.NULL))


# ---- Tiny model fixtures ----

@pytest.fixture
def tiny_model():
    """4->4->2 model, 30 params, microseconds per forward."""
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    )


@pytest.fixture
def tiny_optimizer(tiny_model):
    """SGD optimizer for tiny_model."""
    return SGD(tiny_model.parameters(), lr=0.01)


@pytest.fixture
def tiny_scheduler(tiny_optimizer):
    """StepLR scheduler for tiny_optimizer."""
    return StepLR(tiny_optimizer, step_size=1)


@pytest.fixture
def tiny_batch():
    """Batch of 2 samples, 4 features."""
    torch.manual_seed(42)
    return torch.randn(2, 4)


@pytest.fixture
def tiny_loss_fn():
    """Simple loss: sum of model output."""
    def loss_fn(model, batch):
        return model(batch).sum()
    return loss_fn


@pytest.fixture
def base_config():
    """BaseConfig with defaults."""
    return BaseConfig()


# ---- Component bundle fixtures ----

@pytest.fixture
def backprop_components(tiny_model, tiny_optimizer, tiny_scheduler, tiny_loss_fn, tiny_batch):
    """BackpropComponents wired with tiny model/optimizer/scheduler."""
    return BackpropComponents(
        model=tiny_model,
        optimizer=tiny_optimizer,
        scheduler=tiny_scheduler,
        criterion=None,
        loss_fn=tiny_loss_fn,
        strategy=None,
        data=[tiny_batch],
    )


# ---- Evolutionary fixtures ----

@pytest.fixture
def tiny_fitness_fn():
    """Fitness function: negative sum of all parameters (maximization)."""
    def fitness_fn(model, data):
        return -sum(p.sum().item() for p in model.parameters())
    return fitness_fn


@pytest.fixture
def evolutionary_components(tiny_model, tiny_fitness_fn, tiny_batch):
    """EvolutionaryComponents with no optimizer."""
    return EvolutionaryComponents(
        model=tiny_model,
        fitness_fn=tiny_fitness_fn,
        data=[tiny_batch],
    )


# ---- Adversarial fixtures ----

@pytest.fixture
def adversarial_components():
    """AdversarialComponents with tiny generator and discriminator."""
    torch.manual_seed(42)
    generator = nn.Linear(4, 4)
    discriminator = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 1))
    return AdversarialComponents(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=SGD(generator.parameters(), lr=0.01),
        d_optimizer=SGD(discriminator.parameters(), lr=0.01),
        g_scheduler=None,
        d_scheduler=None,
        strategy=None,
        data=None,
    )


# ---- RL fixtures ----

@pytest.fixture
def rl_components():
    """RLComponents with tiny actor and critic."""
    torch.manual_seed(42)
    actor = nn.Linear(4, 2)
    critic = nn.Linear(4, 1)
    combined_params = list(actor.parameters()) + list(critic.parameters())
    return RLComponents(
        actor=actor,
        critic=critic,
        optimizer=SGD(combined_params, lr=0.01),
        scheduler=None,
        strategy=None,
        data=None,
    )


# ---- Meta-learning fixtures ----

@pytest.fixture
def meta_components():
    """MetaLearningComponents with tiny meta-model."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    return MetaLearningComponents(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        scheduler=None,
        task_loss_fn=lambda m, b: m(b).sum(),
        strategy=None,
        data=None,
    )
