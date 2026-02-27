"""Tests for strategy train_step paths: contrastive data iteration, transformer outputs,
accumulation, and gradient-aligned train_step."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from framework.config import BaseConfig
from framework.strategies.contrastive import ContrastiveTrainStep
from framework.strategies.gradient_aligned_step import GradientAlignedStep, FixedTargetStep
from framework.strategies.strategy_runner import StepResult


# ---- Helpers ----

@dataclass
class ContrastiveTestConfig(BaseConfig):
    accumulation_steps: int = 1


class SimpleEncoder(nn.Module):
    """Tiny encoder that returns raw tensor output."""
    def __init__(self, in_dim=4, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class TransformerStyleEncoder(nn.Module):
    """Encoder that returns objects with .last_hidden_state or .logits."""
    def __init__(self, in_dim=4, hidden_dim=4, output_attr='last_hidden_state'):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self._output_attr = output_attr

    def forward(self, x):
        h = self.linear(x)
        result = SimpleNamespace()
        if self._output_attr == 'last_hidden_state':
            # Shape: [batch, seq_len=1, hidden]
            result.last_hidden_state = h.unsqueeze(1)
        elif self._output_attr == 'logits':
            result.logits = h
        return result


def make_contrastive_components(model, *, accumulation_steps=1, data=None):
    """Build mock components for ContrastiveTrainStep."""
    components = MagicMock()
    components.model = model
    components.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    components.grad_scaler = None
    components.data = data
    components.clip_gradients = MagicMock()
    return components


def simple_augment(batch):
    """Augment by adding small noise."""
    return batch + 0.01 * torch.randn_like(batch), batch + 0.01 * torch.randn_like(batch)


# ---- TestContrastiveDataIteration ----

class TestContrastiveDataIteration:

    def test_train_step_pulls_from_data_when_batch_none(self):
        """When batch=None and self.data exists, train_step pulls from data."""
        encoder = SimpleEncoder(4, 4)
        proj = nn.Linear(4, 4)
        data_batches = [torch.randn(4, 4), torch.randn(4, 4)]
        components = make_contrastive_components(encoder, data=data_batches)

        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=ContrastiveTestConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        result = strategy.train_step(step=1, batch=None)
        assert result.trained is True
        assert 'loss' in result.metrics

    def test_stop_iteration_when_data_exhausted(self):
        """When data is exhausted, train_step returns should_stop=True."""
        encoder = SimpleEncoder(4, 4)
        proj = nn.Linear(4, 4)
        data_batches = [torch.randn(4, 4)]  # only one batch
        components = make_contrastive_components(encoder, data=data_batches)

        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=ContrastiveTestConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        # First call succeeds
        result1 = strategy.train_step(step=0, batch=None)
        assert result1.trained is True

        # Second call: data exhausted
        result2 = strategy.train_step(step=1, batch=None)
        assert result2.trained is False
        assert result2.should_stop is True

    def test_raises_when_no_batch_and_no_data(self):
        """When batch=None and no data source, raises ValueError."""
        encoder = SimpleEncoder(4, 4)
        proj = nn.Linear(4, 4)
        components = make_contrastive_components(encoder, data=None)

        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=ContrastiveTestConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        with pytest.raises(ValueError, match="no batch and no data"):
            strategy.train_step(step=0, batch=None)


# ---- TestContrastiveTransformerOutput ----

class TestContrastiveTransformerOutput:

    def test_encode_handles_last_hidden_state(self):
        """_encode handles model outputs with .last_hidden_state (CLS token)."""
        encoder = TransformerStyleEncoder(4, 4, 'last_hidden_state')
        proj = nn.Linear(4, 4)
        components = make_contrastive_components(encoder)

        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=ContrastiveTestConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        x = torch.randn(2, 4)
        z = strategy._encode(x)
        assert z.shape == (2, 4)
        # Verify normalization
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_handles_logits_attribute(self):
        """_encode handles model outputs with .logits attribute."""
        encoder = TransformerStyleEncoder(4, 4, 'logits')
        proj = nn.Linear(4, 4)
        components = make_contrastive_components(encoder)

        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=ContrastiveTestConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        x = torch.randn(2, 4)
        z = strategy._encode(x)
        assert z.shape == (2, 4)


# ---- TestContrastiveAccumulation ----

class TestContrastiveAccumulation:

    def test_accumulation_defers_optimizer_step(self):
        """With accumulation_steps=2, first step has trained=False."""
        encoder = SimpleEncoder(4, 4)
        proj = nn.Linear(4, 4)
        components = make_contrastive_components(encoder, accumulation_steps=2)

        config = ContrastiveTestConfig(accumulation_steps=2)
        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        batch = torch.randn(4, 4)

        # First step: accumulate (no optimizer step)
        result1 = strategy.train_step(step=0, batch=batch)
        assert result1.trained is False

        # Second step: optimizer steps
        result2 = strategy.train_step(step=1, batch=batch)
        assert result2.trained is True

    def test_accumulation_counter_resets(self):
        """After accumulation completes, counter resets for next cycle."""
        encoder = SimpleEncoder(4, 4)
        proj = nn.Linear(4, 4)
        components = make_contrastive_components(encoder, accumulation_steps=2)

        config = ContrastiveTestConfig(accumulation_steps=2)
        strategy = ContrastiveTrainStep()
        strategy.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=simple_augment,
        )

        batch = torch.randn(4, 4)

        # Complete one accumulation cycle
        strategy.train_step(step=0, batch=batch)
        strategy.train_step(step=1, batch=batch)
        assert strategy._accum_count == 0

        # Next cycle starts fresh
        result = strategy.train_step(step=2, batch=batch)
        assert result.trained is False
        assert strategy._accum_count == 1


# ---- TestGradientAlignedTrainStep ----

class TestGradientAlignedTrainStep:

    def test_empty_selection_stops(self):
        """When selector returns empty list, train_step returns should_stop."""

        class MinimalGradAligned(GradientAlignedStep):
            def compute_target_gradient(self, step):
                return None

            @property
            def name(self):
                return "test_grad_aligned"

        strategy = MinimalGradAligned()

        # Mock all the dependencies
        model = SimpleEncoder(4, 4)
        components = MagicMock()
        components.model = model
        components.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        components.grad_scaler = None

        mock_pool = MagicMock()
        mock_pool.sample = MagicMock(return_value=[])

        mock_selector = MagicMock()
        mock_selector.needs_target_grad = False
        mock_selector.candidates_needed = MagicMock(return_value=10)
        mock_selector.select = MagicMock(return_value=[])

        mock_config = MagicMock()
        mock_config.batch_size = 4
        mock_config.candidates_per_step = 10
        mock_config.max_seq_length = 32

        strategy.setup(
            components=components,
            config=mock_config,
            device=torch.device('cpu'),
            data=mock_pool,
            tokenizer=MagicMock(),
            selector=mock_selector,
        )

        result = strategy.train_step(step=0)
        assert result.trained is False
        assert result.should_stop is True
