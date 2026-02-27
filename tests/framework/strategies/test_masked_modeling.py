"""Tests for MaskedModelingTrainStep."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.masked_modeling import MaskedModelingTrainStep

from .conftest import make_backprop_components


class TestMaskedModelingTrainStep:
    """Tests for MaskedModelingTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = MaskedModelingTrainStep()
        assert step.name == "MaskedModelingTrainStep"

    def test_setup_stores_fields(self):
        """setup() stores all provided references."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)
        mask_fn = lambda b: (b, torch.ones_like(b, dtype=torch.bool), b)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_fn=mask_fn,
            mask_ratio=0.3,
            mask_value=-1.0,
        )

        assert step.mask_fn is mask_fn
        assert step.mask_ratio == 0.3
        assert step.mask_value == -1.0

    def test_setup_defaults(self):
        """setup() uses default mask_ratio=0.15 and mask_value=0.0."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.mask_ratio == 0.15
        assert step.mask_value == 0.0
        assert step.mask_fn is None

    def test_default_mask_ratio(self):
        """_default_mask masks approximately the target fraction."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_ratio=0.5,
        )

        batch = torch.randn(100, 4)
        _, mask, _ = step._default_mask(batch)
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - 0.5) < 0.1

    def test_default_mask_preserves_unmasked(self):
        """Unmasked positions retain their original values."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_ratio=0.5,
        )

        batch = torch.randn(4, 4)
        masked_batch, mask, targets = step._default_mask(batch)
        unmasked = ~mask
        assert torch.allclose(masked_batch[unmasked], batch[unmasked])

    def test_default_mask_replaces_masked(self):
        """Masked positions are replaced with mask_value."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_ratio=0.5,
            mask_value=-99.0,
        )

        batch = torch.randn(4, 4) + 10  # shift away from -99
        masked_batch, mask, _ = step._default_mask(batch)
        if mask.any():
            assert (masked_batch[mask] == -99.0).all()

    def test_custom_mask_fn_used(self):
        """When mask_fn is provided, it is called instead of default."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        calls = []

        def custom_mask(batch):
            calls.append(1)
            mask = torch.zeros_like(batch, dtype=torch.bool)
            mask[:, 0] = True  # mask first feature
            masked = batch.clone()
            masked[mask] = 0.0
            return masked, mask, batch

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_fn=custom_mask,
        )

        batch = torch.randn(2, 4)
        step.train_step(step=1, batch=batch)
        assert len(calls) == 1

    def test_compute_masked_loss_ignores_unmasked(self):
        """Loss is computed only on masked positions."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        predictions = torch.randn(2, 4)
        targets = torch.randn(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[0, 0] = True  # only one position masked

        loss = step._compute_masked_loss(predictions, targets, mask)
        expected = F.mse_loss(predictions[0, 0:1], targets[0, 0:1])
        assert torch.allclose(loss, expected)

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            mask_ratio=0.5,
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'mask_ratio_actual' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model, data=None)

        step = MaskedModelingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)
