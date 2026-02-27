"""Tests for SelfDistillationTrainStep."""

import pytest
import torch
import torch.nn as nn

from framework.config import BaseConfig
from framework.strategies.strategy_runner import StepResult
from framework.strategies.self_distillation import SelfDistillationTrainStep

from .conftest import make_backprop_components


class TestSelfDistillationTrainStep:
    """Tests for SelfDistillationTrainStep."""

    def _make_components(self):
        """Create components with EMA encoder in auxiliary_models."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        ema_encoder = nn.Linear(4, 4)
        ema_encoder.load_state_dict(encoder.state_dict())
        components = make_backprop_components(
            encoder, auxiliary_models={'ema_encoder': ema_encoder}
        )
        return components, ema_encoder

    def _setup_step(self, components, **extra_kwargs):
        """Create and setup a SelfDistillationTrainStep with defaults."""
        kwargs = {
            'projection_head': nn.Linear(4, 4),
            'predictor': nn.Linear(4, 4),
            'augment_fn': lambda x: (x, x),
            'ema_projection_head': nn.Linear(4, 4),
        }
        kwargs.update(extra_kwargs)

        step = SelfDistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            **kwargs,
        )
        return step

    def test_name(self):
        """Reports correct name."""
        step = SelfDistillationTrainStep()
        assert step.name == "SelfDistillationTrainStep"

    def test_requires_projection_head(self):
        """setup() raises when projection_head is not provided."""
        components, _ = self._make_components()
        step = SelfDistillationTrainStep()
        with pytest.raises(ValueError, match="requires a projection_head"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                predictor=nn.Linear(4, 4),
                augment_fn=lambda x: (x, x),
                ema_projection_head=nn.Linear(4, 4),
            )

    def test_requires_predictor(self):
        """setup() raises when predictor is not provided."""
        components, _ = self._make_components()
        step = SelfDistillationTrainStep()
        with pytest.raises(ValueError, match="requires a predictor"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=nn.Linear(4, 4),
                augment_fn=lambda x: (x, x),
                ema_projection_head=nn.Linear(4, 4),
            )

    def test_requires_augment_fn(self):
        """setup() raises when augment_fn is not provided."""
        components, _ = self._make_components()
        step = SelfDistillationTrainStep()
        with pytest.raises(ValueError, match="requires an augment_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=nn.Linear(4, 4),
                predictor=nn.Linear(4, 4),
                ema_projection_head=nn.Linear(4, 4),
            )

    def test_requires_ema_encoder(self):
        """setup() raises when auxiliary_models lacks 'ema_encoder'."""
        model = nn.Linear(4, 4)
        components = make_backprop_components(model, auxiliary_models={})
        step = SelfDistillationTrainStep()
        with pytest.raises(ValueError, match="requires auxiliary_models"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=nn.Linear(4, 4),
                predictor=nn.Linear(4, 4),
                augment_fn=lambda x: (x, x),
                ema_projection_head=nn.Linear(4, 4),
            )

    def test_requires_ema_projection_head(self):
        """setup() raises when ema_projection_head is not provided."""
        components, _ = self._make_components()
        step = SelfDistillationTrainStep()
        with pytest.raises(ValueError, match="requires kwargs"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=nn.Linear(4, 4),
                predictor=nn.Linear(4, 4),
                augment_fn=lambda x: (x, x),
            )

    def test_setup_stores_fields(self):
        """setup() stores all provided references."""
        components, ema_encoder = self._make_components()
        proj = nn.Linear(4, 4)
        pred = nn.Linear(4, 4)
        ema_proj = nn.Linear(4, 4)
        augment = lambda x: (x, x)

        step = SelfDistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            predictor=pred,
            augment_fn=augment,
            ema_projection_head=ema_proj,
            ema_decay=0.99,
            loss_type='mse',
            symmetrize=False,
        )

        assert step.projection_head is proj
        assert step.predictor is pred
        assert step.augment_fn is augment
        assert step.ema_encoder is ema_encoder
        assert step.ema_projection_head is ema_proj
        assert step.ema_decay == 0.99
        assert step.loss_type == 'mse'
        assert step.symmetrize is False

    def test_setup_defaults(self):
        """setup() uses default ema_decay=0.996, loss_type='cosine', symmetrize=True."""
        components, _ = self._make_components()
        step = self._setup_step(components)

        assert step.ema_decay == 0.996
        assert step.loss_type == 'cosine'
        assert step.symmetrize is True

    def test_ema_encoder_set_to_eval(self):
        """EMA encoder is put in eval mode during setup."""
        components, ema_encoder = self._make_components()
        ema_encoder.train()  # force train mode
        self._setup_step(components)
        assert not ema_encoder.training

    def test_cosine_loss_bounded(self):
        """Cosine loss is bounded in [0, 4]."""
        components, _ = self._make_components()
        step = self._setup_step(components)

        z1 = torch.randn(3, 4)
        z2 = torch.randn(3, 4)
        loss = step._cosine_loss(z1, z2)
        assert 0.0 <= loss.item() <= 4.0

    def test_vicreg_loss_components(self):
        """VICReg loss returns finite sim, var, cov components."""
        components, _ = self._make_components()
        step = self._setup_step(components, loss_type='vicreg')

        z1 = torch.randn(4, 4)
        z2 = torch.randn(4, 4)
        total, sim, var, cov = step._vicreg_loss(z1, z2)

        assert torch.isfinite(total)
        assert torch.isfinite(sim)
        assert torch.isfinite(var)
        assert torch.isfinite(cov)

    def test_update_momentum_ema(self):
        """_update_momentum moves target params toward online params."""
        torch.manual_seed(42)
        components, ema_encoder = self._make_components()
        step = self._setup_step(components, ema_decay=0.5)

        online_params = [p.clone() for p in components.model.parameters()]
        target_params_before = [p.clone() for p in ema_encoder.parameters()]

        step._update_momentum()

        for online_p, before_p, after_p in zip(
            online_params, target_params_before, ema_encoder.parameters()
        ):
            expected = 0.5 * before_p + 0.5 * online_p
            assert torch.allclose(after_p.data, expected, atol=1e-6)

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        components, _ = self._make_components()
        step = self._setup_step(components)

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        ema_encoder = nn.Linear(4, 4)
        components = make_backprop_components(
            encoder, data=None, auxiliary_models={'ema_encoder': ema_encoder}
        )
        step = self._setup_step(components)

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)
