"""Tests for MultiTaskTrainStep."""

import pytest
import torch
import torch.nn as nn

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.multi_task import MultiTaskTrainStep

from .conftest import make_backprop_components


def _make_task_loss_fns():
    """Two simple task loss functions for a Linear(4, 4) model."""
    return {
        'task_a': lambda m, b: m(b)[:, :2].sum(),
        'task_b': lambda m, b: m(b)[:, 2:].sum(),
    }


class TestMultiTaskTrainStep:
    """Tests for MultiTaskTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = MultiTaskTrainStep()
        assert step.name == "MultiTaskTrainStep"

    def test_requires_task_loss_fns(self):
        """setup() raises when task_loss_fns is not provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)
        step = MultiTaskTrainStep()
        with pytest.raises(ValueError, match="requires task_loss_fns"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_requires_at_least_two_tasks(self):
        """setup() raises when fewer than 2 tasks provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)
        step = MultiTaskTrainStep()
        with pytest.raises(ValueError, match="at least 2 tasks"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                task_loss_fns={'only_one': lambda m, b: m(b).sum()},
            )

    def test_setup_stores_fields(self):
        """setup() stores task_loss_fns and weighting."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)
        task_fns = _make_task_loss_fns()

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=task_fns,
            weighting='uniform',
        )

        assert step.task_loss_fns is task_fns
        assert step.weighting == 'uniform'
        assert step.task_names == ['task_a', 'task_b']

    def test_setup_defaults(self):
        """setup() uses default weighting='uncertainty'."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
        )

        assert step.weighting == 'uncertainty'

    def test_uncertainty_creates_log_vars(self):
        """Uncertainty weighting creates log_var parameters per task."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
        )

        assert hasattr(step, 'log_vars')
        assert 'task_a' in step.log_vars
        assert 'task_b' in step.log_vars
        # Parameters should be initialized to 0
        assert step.log_vars['task_a'].item() == 0.0
        assert step.log_vars['task_b'].item() == 0.0

    def test_uniform_weighting(self):
        """Uniform weighting gives equal weight to all tasks."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
            weighting='uniform',
        )

        weights = step.get_task_weights()
        assert weights['task_a'] == pytest.approx(0.5)
        assert weights['task_b'] == pytest.approx(0.5)

    def test_fixed_weighting(self):
        """Fixed weighting uses provided weights."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
            weighting='fixed',
            fixed_weights={'task_a': 0.3, 'task_b': 0.7},
        )

        weights = step.get_task_weights()
        assert weights['task_a'] == 0.3
        assert weights['task_b'] == 0.7

    def test_fixed_requires_weights(self):
        """setup() raises when weighting='fixed' but no fixed_weights."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)
        step = MultiTaskTrainStep()
        with pytest.raises(ValueError, match="requires fixed_weights"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                task_loss_fns=_make_task_loss_fns(),
                weighting='fixed',
            )

    def test_get_task_weights_uncertainty(self):
        """Uncertainty weights are positive (exp(-log_var) > 0)."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
        )

        weights = step.get_task_weights()
        assert weights['task_a'] > 0
        assert weights['task_b'] > 0

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_reports_per_task_metrics(self):
        """Metrics include per-task loss and weight."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
            weighting='uniform',
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert 'task_task_a_loss' in result.metrics
        assert 'task_task_b_loss' in result.metrics
        assert 'task_task_a_weight' in result.metrics
        assert 'task_task_b_weight' in result.metrics

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = make_backprop_components(model, data=None)

        step = MultiTaskTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            task_loss_fns=_make_task_loss_fns(),
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)
