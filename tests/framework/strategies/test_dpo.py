"""Tests for DPOTrainStep."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.dpo import DPOTrainStep

from .conftest import make_backprop_components


def _log_prob_fn(model, batch):
    """Simple log prob: log_softmax of model output, summed over features."""
    return F.log_softmax(model(batch), dim=-1).sum(-1)


class TestDPOTrainStep:
    """Tests for DPOTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = DPOTrainStep()
        assert step.name == "DPOTrainStep"

    def test_requires_reference_model(self):
        """setup() raises when auxiliary_models lacks 'reference_model'."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        components = make_backprop_components(model, auxiliary_models={})
        step = DPOTrainStep()
        with pytest.raises(ValueError, match="requires auxiliary_models"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                log_prob_fn=_log_prob_fn,
            )

    def test_requires_log_prob_fn(self):
        """setup() raises when log_prob_fn is not provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )
        step = DPOTrainStep()
        with pytest.raises(ValueError, match="requires a log_prob_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_stores_fields(self):
        """setup() stores all provided references."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
            beta=0.2,
            label_smoothing=0.1,
        )

        assert step.reference_model is ref
        assert step.log_prob_fn is _log_prob_fn
        assert step.beta == 0.2
        assert step.label_smoothing == 0.1

    def test_setup_defaults(self):
        """setup() uses default beta=0.1 and label_smoothing=0.0."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        assert step.beta == 0.1
        assert step.label_smoothing == 0.0

    def test_reference_model_set_to_eval(self):
        """Reference model is put in eval mode during setup."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        ref.train()
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        assert not ref.training

    def test_dpo_loss_basic(self):
        """DPO loss returns finite value for known log probs."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        pi_c = torch.tensor([1.0, 0.5])
        pi_r = torch.tensor([0.5, 0.3])
        ref_c = torch.tensor([0.8, 0.4])
        ref_r = torch.tensor([0.6, 0.4])

        loss, chosen_r, rejected_r = step._dpo_loss(pi_c, pi_r, ref_c, ref_r)
        assert torch.isfinite(loss)

    def test_dpo_loss_prefers_chosen(self):
        """When policy strongly prefers chosen, loss should be low."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
            beta=1.0,
        )

        # Policy strongly prefers chosen over rejected
        pi_c = torch.tensor([5.0])
        pi_r = torch.tensor([-5.0])
        ref_c = torch.tensor([0.0])
        ref_r = torch.tensor([0.0])

        loss_good, _, _ = step._dpo_loss(pi_c, pi_r, ref_c, ref_r)

        # Policy prefers rejected
        pi_c2 = torch.tensor([-5.0])
        pi_r2 = torch.tensor([5.0])

        loss_bad, _, _ = step._dpo_loss(pi_c2, pi_r2, ref_c, ref_r)

        assert loss_good.item() < loss_bad.item()

    def test_unpack_batch_tuple(self):
        """_unpack_batch handles tuple input."""
        chosen = torch.randn(2, 4)
        rejected = torch.randn(2, 4)
        c, r = DPOTrainStep._unpack_batch((chosen, rejected))
        assert c is chosen
        assert r is rejected

    def test_unpack_batch_object(self):
        """_unpack_batch handles object with .chosen/.rejected."""

        class PreferenceBatch:
            def __init__(self, chosen, rejected):
                self.chosen = chosen
                self.rejected = rejected

        chosen = torch.randn(2, 4)
        rejected = torch.randn(2, 4)
        batch = PreferenceBatch(chosen, rejected)
        c, r = DPOTrainStep._unpack_batch(batch)
        assert c is chosen
        assert r is rejected

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult with all metrics."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        chosen = torch.randn(2, 4)
        rejected = torch.randn(2, 4)
        result = step.train_step(step=1, batch=(chosen, rejected))

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'chosen_reward' in result.metrics
        assert 'rejected_reward' in result.metrics
        assert 'reward_margin' in result.metrics
        assert 'accuracy' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        components = make_backprop_components(
            model, data=None, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)

    def test_reward_accuracy_metric(self):
        """Accuracy reflects correct preference rankings."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        ref = nn.Linear(4, 2)
        # Make ref identical so rewards are purely from policy
        ref.load_state_dict(model.state_dict())
        components = make_backprop_components(
            model, auxiliary_models={'reference_model': ref}
        )

        step = DPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            log_prob_fn=_log_prob_fn,
        )

        chosen = torch.randn(4, 4)
        rejected = torch.randn(4, 4)
        result = step.train_step(step=1, batch=(chosen, rejected))

        # Accuracy should be between 0 and 1
        assert 0.0 <= result.metrics['accuracy'] <= 1.0
