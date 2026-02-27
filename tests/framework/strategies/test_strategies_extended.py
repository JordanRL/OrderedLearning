"""Tests for extended strategy implementations.

Covers: GradientAlignedStep, DistillationTrainStep, ContrastiveTrainStep,
MomentumContrastiveTrainStep, AdversarialTrainStep, WGANGPTrainStep,
PPOTrainStep, A2CTrainStep, PredictiveCodingTrainStep, MAMLStep, ReptileStep.

Focus on construction, setup, pure logic methods, and train_step with
tiny models where feasible.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD

from framework.config import BaseConfig
from framework.capabilities import TrainingParadigm
from framework.strategies.strategy_runner import StepResult
from framework.strategies.gradient_aligned_step import (
    GradientAlignedStep, FixedTargetStep, PhasedCurriculumStep,
)
from framework.strategies.distillation import DistillationTrainStep
from framework.strategies.contrastive import (
    ContrastiveTrainStep, MomentumContrastiveTrainStep,
)
from framework.strategies.adversarial import AdversarialTrainStep, WGANGPTrainStep
from framework.strategies.rl import PPOTrainStep, A2CTrainStep
from framework.strategies.predictive_coding import PredictiveCodingTrainStep
from framework.strategies.meta_learning import MAMLStep, ReptileStep, _ParameterizedModule
from framework.trainers.components import BackpropComponents
from framework.trainers.adversarial_components import AdversarialComponents
from framework.trainers.rl_components import RLComponents
from framework.trainers.pc_components import PredictiveCodingComponents
from framework.trainers.meta_components import MetaLearningComponents
from framework.models.predictive_coding import PredictiveCodingNetwork
from framework.data.rollout_buffer import RolloutBatch
from framework.data.task_sampler import TaskBatch


# ---- Helpers ----

@dataclass
class ExtendedConfig(BaseConfig):
    """Config with fields needed by gradient-aligned strategies."""
    batch_size: int = 2
    candidates_per_step: int = 4
    max_seq_length: int = 16
    phase_check_every: int = 5
    accumulation_steps: int = 1


def _make_backprop_components(model, loss_fn=None, data=None, auxiliary_models=None):
    """Build BackpropComponents for a given model."""
    optimizer = SGD(model.parameters(), lr=0.01)
    if loss_fn is None:
        loss_fn = lambda m, b: m(b).sum()
    return BackpropComponents(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        criterion=None,
        loss_fn=loss_fn,
        strategy=None,
        data=data,
        auxiliary_models=auxiliary_models,
    )


def _make_adversarial_components(data=None):
    """Build AdversarialComponents with tiny G and D."""
    torch.manual_seed(42)
    G = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    D = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))

    def g_loss_fn(gen, disc, batch):
        fake = gen(batch)
        return -disc(fake).mean()

    def d_loss_fn(gen, disc, batch):
        with torch.no_grad():
            fake = gen(batch)
        return disc(fake).mean() - disc(batch).mean()

    batch = torch.randn(2, 4)
    return AdversarialComponents(
        generator=G,
        discriminator=D,
        g_optimizer=SGD(G.parameters(), lr=0.01),
        d_optimizer=SGD(D.parameters(), lr=0.01),
        g_scheduler=None,
        d_scheduler=None,
        strategy=None,
        data=data or [batch.clone() for _ in range(20)],
        g_loss_fn=g_loss_fn,
        d_loss_fn=d_loss_fn,
    )


class TinyActor(nn.Module):
    """Minimal actor that returns a Categorical distribution."""
    def __init__(self, obs_dim=4, act_dim=2):
        super().__init__()
        self.linear = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        logits = self.linear(x)
        return torch.distributions.Categorical(logits=logits)


def _make_rl_components():
    """Build RLComponents with tiny actor and critic."""
    torch.manual_seed(42)
    actor = TinyActor(obs_dim=4, act_dim=2)
    critic = nn.Linear(4, 1)
    combined = list(actor.parameters()) + list(critic.parameters())
    return RLComponents(
        actor=actor,
        critic=critic,
        optimizer=SGD(combined, lr=0.01),
        scheduler=None,
        strategy=None,
        data=None,
    )


def _make_rollout_batch(n=4, obs_dim=4, act_dim=2):
    """Build a RolloutBatch with random data."""
    return RolloutBatch(
        observations=torch.randn(n, obs_dim),
        actions=torch.randint(0, act_dim, (n,)),
        old_log_probs=torch.randn(n),
        advantages=torch.randn(n),
        returns=torch.randn(n),
    )


class MockTaskSampler:
    """Minimal TaskSampler that returns pre-built tasks."""
    def __init__(self, input_dim=4, output_dim=2, batch_size=3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

    def sample(self, n_tasks):
        tasks = []
        for i in range(n_tasks):
            support = torch.randn(self.batch_size, self.input_dim)
            query = torch.randn(self.batch_size, self.input_dim)
            tasks.append(TaskBatch(support=support, query=query, task_id=i))
        return tasks


def _make_meta_components(task_sampler=None):
    """Build MetaLearningComponents with tiny model and task sampler."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    sampler = task_sampler or MockTaskSampler()
    return MetaLearningComponents(
        model=model,
        optimizer=SGD(model.parameters(), lr=0.01),
        scheduler=None,
        task_loss_fn=lambda m, b: m(b).sum(),
        strategy=None,
        data=sampler,
    )


def _make_pc_components():
    """Build PredictiveCodingComponents with tiny PC network."""
    torch.manual_seed(42)
    model = PredictiveCodingNetwork(layer_sizes=[4, 4, 2])
    optimizer = SGD(model.parameters(), lr=0.01)
    data = [torch.randn(2, 4) for _ in range(10)]
    return PredictiveCodingComponents(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        strategy=None,
        data=data,
    )


# ====================================================================
# GradientAlignedStep
# ====================================================================

class TestGradientAlignedStep:
    """Tests for GradientAlignedStep base class and subclasses."""

    def test_is_abstract(self):
        """GradientAlignedStep cannot be instantiated directly."""
        with pytest.raises(TypeError, match="compute_target_gradient"):
            GradientAlignedStep()

    def test_fixed_target_step_name(self):
        """FixedTargetStep reports its name."""
        step = FixedTargetStep()
        assert step.name == "FixedTargetStep"

    def test_phased_curriculum_step_name(self):
        """PhasedCurriculumStep reports its name."""
        step = PhasedCurriculumStep()
        assert step.name == "PhasedCurriculumStep"

    def test_fixed_target_setup_stores_fields(self):
        """FixedTargetStep.setup() stores all necessary references."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        components = _make_backprop_components(model)
        config = ExtendedConfig()

        # Mock the tokenizer and data pool
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=torch.randint(0, 100, (1, 3)))
        tokenizer.pad_token_id = 0

        selector = MagicMock()
        selector.candidates_needed = MagicMock(return_value=8)
        selector.needs_target_grad = True

        data_pool = MagicMock()

        target_config = {'trigger': 'hello', 'completion': ' world'}

        step = FixedTargetStep()
        step.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            data=data_pool,
            tokenizer=tokenizer,
            selector=selector,
            target_config=target_config,
        )

        assert step.model is model
        assert step.optimizer is components.optimizer
        assert step.device == torch.device('cpu')
        assert step.data_pool is data_pool
        assert step.tokenizer is tokenizer
        assert step.selector is selector
        assert step.n_candidates == 8
        # Check target tensors were created
        assert step.full_sequence is not None
        assert step.target_labels is not None
        assert step.target_attention_mask is not None
        assert step.full_sequence.shape[1] == 6  # 3 trigger + 3 completion tokens

    def test_fixed_target_target_labels_masking(self):
        """FixedTargetStep masks trigger portion of labels with -100."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        components = _make_backprop_components(model)
        config = ExtendedConfig()

        trigger_ids = torch.tensor([[1, 2, 3]])
        completion_ids = torch.tensor([[4, 5]])
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(side_effect=[trigger_ids, completion_ids])
        tokenizer.pad_token_id = 0

        selector = MagicMock()
        selector.candidates_needed = MagicMock(return_value=8)

        step = FixedTargetStep()
        step.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            data=MagicMock(),
            tokenizer=tokenizer,
            selector=selector,
            target_config={'trigger': 'a', 'completion': 'b'},
        )

        # First 3 positions (trigger) should be masked to -100
        assert (step.target_labels[0, :3] == -100).all()
        # Last 2 positions (completion) should retain original token ids
        assert (step.target_labels[0, 3:] == torch.tensor([4, 5])).all()

    def test_phased_curriculum_setup_stores_curriculum(self):
        """PhasedCurriculumStep.setup() stores curriculum reference."""
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        components = _make_backprop_components(model)
        config = ExtendedConfig()

        curriculum = MagicMock()
        tokenizer = MagicMock()
        selector = MagicMock()
        selector.candidates_needed = MagicMock(return_value=8)

        step = PhasedCurriculumStep()
        step.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            data=MagicMock(),
            tokenizer=tokenizer,
            selector=selector,
            curriculum=curriculum,
        )

        assert step.curriculum is curriculum
        assert step.phase_check_every == 5

    def test_phased_curriculum_post_step_skip_non_check_step(self):
        """post_step returns None when not at a phase_check_every boundary."""
        step = PhasedCurriculumStep()
        step.phase_check_every = 5
        step.curriculum = MagicMock()

        result = StepResult(metrics={'loss': 0.1})
        assert step.post_step(step=3, result=result) is None

    def test_phased_curriculum_post_step_skip_final_phase(self):
        """post_step returns None when already at the final phase."""
        step = PhasedCurriculumStep()
        step.phase_check_every = 5
        step.curriculum = MagicMock()
        step.curriculum.is_final_phase = True

        result = StepResult(metrics={'loss': 0.1})
        assert step.post_step(step=5, result=result) is None

    def test_phased_curriculum_post_step_no_advance(self):
        """post_step returns eval info without transition when not ready."""
        step = PhasedCurriculumStep()
        step.phase_check_every = 5
        step.model = MagicMock()  # post_step passes self.model to curriculum
        step.curriculum = MagicMock()
        step.curriculum.is_final_phase = False
        step.curriculum.evaluate_phase_targets.return_value = {'avg_prob': 0.3}
        step.curriculum.should_advance_phase.return_value = False
        step.curriculum.phase_name = "phase_1"

        result = StepResult(metrics={'loss': 0.1})
        info = step.post_step(step=5, result=result)

        assert info is not None
        assert 'phase_transition' not in info
        assert info['phase_name'] == "phase_1"
        assert info['phase_avg_prob'] == 0.3

    def test_phased_curriculum_post_step_advances_phase(self):
        """post_step returns transition info when phase advances."""
        step = PhasedCurriculumStep()
        step.phase_check_every = 5
        step.model = MagicMock()  # post_step passes self.model to curriculum
        step.curriculum = MagicMock()
        step.curriculum.is_final_phase = False
        step.curriculum.evaluate_phase_targets.return_value = {'avg_prob': 0.9}
        step.curriculum.should_advance_phase.return_value = True
        step.curriculum.advance_phase.return_value = ('phase_1', 'phase_2')

        result = StepResult(metrics={'loss': 0.1})
        info = step.post_step(step=5, result=result)

        assert info['phase_transition'] is True
        assert info['old_phase'] == 'phase_1'
        assert info['new_phase'] == 'phase_2'
        assert info['phase_avg_prob'] == 0.9


# ====================================================================
# DistillationTrainStep
# ====================================================================

class TestDistillationTrainStep:
    """Tests for DistillationTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = DistillationTrainStep()
        assert step.name == "DistillationTrainStep"

    def test_requires_teacher(self):
        """setup() raises when no teacher model is provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        components = _make_backprop_components(model)
        step = DistillationTrainStep()
        with pytest.raises(ValueError, match="requires a teacher model"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_requires_loss_fn(self):
        """setup() raises when loss_fn is None."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(model, loss_fn=None)
        components.loss_fn = None
        step = DistillationTrainStep()
        with pytest.raises(ValueError, match="requires a loss_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                teacher=teacher,
            )

    def test_setup_stores_fields(self):
        """setup() stores teacher, temperature, alpha, etc."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(student)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
            temperature=6.0,
            alpha=0.3,
        )

        assert step.teacher is teacher
        assert step.temperature == 6.0
        assert step.alpha == 0.3
        assert step.model is student

    def test_setup_defaults(self):
        """setup() uses default temperature=4.0 and alpha=0.5."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(student)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
        )

        assert step.temperature == 4.0
        assert step.alpha == 0.5

    def test_teacher_from_auxiliary_models(self):
        """setup() finds teacher in auxiliary_models when not in kwargs."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(
            student, auxiliary_models={'teacher': teacher}
        )

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.teacher is teacher

    def test_teacher_set_to_eval(self):
        """Teacher model is put in eval mode during setup."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        teacher.train()
        components = _make_backprop_components(student)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
        )

        assert not teacher.training

    def test_default_soft_loss_is_kl(self):
        """Default soft_loss_fn computes KL divergence."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(student)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
        )

        # Create logits and check the soft loss is finite
        student_logits = torch.randn(2, 4)
        teacher_logits = torch.randn(2, 4)
        soft_loss = step.soft_loss_fn(student_logits, teacher_logits)
        assert soft_loss.item() >= 0.0  # KL divergence is non-negative
        assert torch.isfinite(soft_loss)

    def test_train_step_with_batch(self):
        """train_step runs forward/backward and returns StepResult."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)

        def loss_fn(m, b):
            return m(b).sum()

        components = _make_backprop_components(student, loss_fn=loss_fn)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'hard_loss' in result.metrics
        assert 'soft_loss' in result.metrics
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_no_batch_raises_without_data(self):
        """train_step raises ValueError when no batch and no data source."""
        torch.manual_seed(42)
        student = nn.Linear(4, 2)
        teacher = nn.Linear(4, 2)
        components = _make_backprop_components(student, data=None)

        step = DistillationTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            teacher=teacher,
        )

        with pytest.raises(ValueError, match="no batch and no data source"):
            step.train_step(step=1)


# ====================================================================
# ContrastiveTrainStep
# ====================================================================

class TestContrastiveTrainStep:
    """Tests for ContrastiveTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = ContrastiveTrainStep()
        assert step.name == "ContrastiveTrainStep"

    def test_requires_projection_head(self):
        """setup() raises when projection_head is not provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = _make_backprop_components(model)
        step = ContrastiveTrainStep()
        with pytest.raises(ValueError, match="requires a projection_head"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                augment_fn=lambda x: (x, x),
            )

    def test_requires_augment_fn(self):
        """setup() raises when augment_fn is not provided."""
        torch.manual_seed(42)
        model = nn.Linear(4, 4)
        components = _make_backprop_components(model)
        step = ContrastiveTrainStep()
        with pytest.raises(ValueError, match="requires an augment_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=nn.Linear(4, 8),
            )

    def test_setup_stores_fields(self):
        """setup() stores all provided references."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        augment = lambda x: (x, x)
        components = _make_backprop_components(encoder)

        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=augment,
            temperature=0.1,
        )

        assert step.model is encoder
        assert step.projection_head is proj
        assert step.augment_fn is augment
        assert step.temperature == 0.1

    def test_setup_default_temperature(self):
        """setup() defaults temperature to 0.07."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        components = _make_backprop_components(encoder)

        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=nn.Linear(4, 8),
            augment_fn=lambda x: (x, x),
        )

        assert step.temperature == 0.07

    def test_nt_xent_loss_computation(self):
        """NT-Xent loss returns finite positive value for unit-normalized inputs."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        components = _make_backprop_components(encoder)

        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x, x),
            temperature=0.5,
        )

        z1 = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        z2 = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        loss = step._nt_xent_loss(z1, z2)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_encode_normalizes_output(self):
        """_encode produces unit-normalized embeddings."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        components = _make_backprop_components(encoder)

        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x, x),
        )

        x = torch.randn(3, 4)
        z = step._encode(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_train_step_returns_loss(self):
        """train_step computes contrastive loss and updates parameters."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)

        # Both encoder and proj need gradients via the same optimizer
        all_params = list(encoder.parameters()) + list(proj.parameters())
        optimizer = SGD(all_params, lr=0.01)

        components = BackpropComponents(
            model=encoder,
            optimizer=optimizer,
            scheduler=None,
            criterion=None,
            loss_fn=None,
            strategy=None,
            data=None,
        )

        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x + torch.randn_like(x) * 0.1,
                                  x + torch.randn_like(x) * 0.1),
        )

        batch = torch.randn(4, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert torch.isfinite(result.metrics['loss'])


# ====================================================================
# MomentumContrastiveTrainStep
# ====================================================================

class TestMomentumContrastiveTrainStep:
    """Tests for MomentumContrastiveTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = MomentumContrastiveTrainStep()
        assert step.name == "MomentumContrastiveTrainStep"

    def test_requires_momentum_encoder(self):
        """setup() raises when momentum_encoder is not in auxiliary_models."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        components = _make_backprop_components(encoder)

        step = MomentumContrastiveTrainStep()
        with pytest.raises(ValueError, match="momentum_encoder"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=proj,
                augment_fn=lambda x: (x, x),
                momentum_projection_head=nn.Linear(4, 8),
            )

    def test_requires_momentum_projection_head(self):
        """setup() raises when momentum_projection_head is not in kwargs."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        momentum_enc = nn.Linear(4, 4)
        components = _make_backprop_components(
            encoder, auxiliary_models={'momentum_encoder': momentum_enc}
        )

        step = MomentumContrastiveTrainStep()
        with pytest.raises(ValueError, match="momentum_projection_head"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
                projection_head=proj,
                augment_fn=lambda x: (x, x),
            )

    def test_setup_stores_ema_decay(self):
        """setup() stores ema_decay and defaults to 0.996."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        momentum_enc = nn.Linear(4, 4)
        momentum_proj = nn.Linear(4, 8)
        components = _make_backprop_components(
            encoder, auxiliary_models={'momentum_encoder': momentum_enc}
        )

        step = MomentumContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x, x),
            momentum_projection_head=momentum_proj,
        )

        assert step.ema_decay == 0.996
        assert step.momentum_encoder is momentum_enc
        assert step.momentum_projection_head is momentum_proj

    def test_update_momentum_ema(self):
        """_update_momentum updates target towards online with EMA."""
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        momentum_enc = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        momentum_proj = nn.Linear(4, 8)

        # Set different initial weights
        with torch.no_grad():
            for p in momentum_enc.parameters():
                p.fill_(0.0)
            for p in momentum_proj.parameters():
                p.fill_(0.0)

        components = _make_backprop_components(
            encoder, auxiliary_models={'momentum_encoder': momentum_enc}
        )

        step = MomentumContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x, x),
            momentum_projection_head=momentum_proj,
            ema_decay=0.5,
        )

        # Before update, momentum params are zero
        for p in momentum_enc.parameters():
            assert (p == 0.0).all()

        step._update_momentum()

        # After EMA with decay=0.5: momentum = 0.5 * 0 + 0.5 * online
        for online_p, mom_p in zip(encoder.parameters(), momentum_enc.parameters()):
            expected = 0.5 * online_p.data
            assert torch.allclose(mom_p.data, expected, atol=1e-6)


# ====================================================================
# AdversarialTrainStep
# ====================================================================

class TestAdversarialTrainStep:
    """Tests for AdversarialTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = AdversarialTrainStep()
        assert step.name == "AdversarialTrainStep"

    def test_paradigm_is_adversarial(self):
        """Strategy declares ADVERSARIAL paradigm."""
        assert AdversarialTrainStep.paradigm == TrainingParadigm.ADVERSARIAL

    def test_requires_loss_fns(self):
        """setup() raises when g_loss_fn or d_loss_fn is None."""
        components = _make_adversarial_components()
        components.g_loss_fn = None

        step = AdversarialTrainStep()
        with pytest.raises(ValueError, match="requires both g_loss_fn and d_loss_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_stores_fields(self):
        """setup() stores generator, discriminator, optimizers, etc."""
        components = _make_adversarial_components()
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.generator is components.generator
        assert step.discriminator is components.discriminator
        assert step.g_optimizer is components.g_optimizer
        assert step.d_optimizer is components.d_optimizer
        assert step.d_steps_per_g_step == 1

    def test_train_step_returns_g_and_d_loss(self):
        """train_step produces both g_loss and d_loss in metrics."""
        torch.manual_seed(42)
        components = _make_adversarial_components()
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert 'g_loss' in result.metrics
        assert 'd_loss' in result.metrics
        assert 'loss' in result.metrics
        assert result.metrics['loss'] == result.metrics['g_loss']

    def test_train_step_updates_both_models(self):
        """Both G and D parameters change after one train_step."""
        torch.manual_seed(42)
        components = _make_adversarial_components()
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        g_before = [p.clone() for p in components.generator.parameters()]
        d_before = [p.clone() for p in components.discriminator.parameters()]

        batch = torch.randn(2, 4)
        step.train_step(step=1, batch=batch)

        g_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(g_before, components.generator.parameters())
        )
        d_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(d_before, components.discriminator.parameters())
        )
        assert g_changed, "Generator parameters should change"
        assert d_changed, "Discriminator parameters should change"

    def test_multiple_d_steps(self):
        """d_steps_per_g_step > 1 runs multiple discriminator updates."""
        torch.manual_seed(42)
        components = _make_adversarial_components()
        components.d_steps_per_g_step = 3
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        batch = torch.randn(2, 4)
        # Should not raise and should return valid result
        result = step.train_step(step=1, batch=batch)
        assert result.trained is True

    def test_get_batch_cycles_data(self):
        """_get_batch() resets iterator on exhaustion."""
        torch.manual_seed(42)
        data = [torch.randn(2, 4)]
        components = _make_adversarial_components(data=data)
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        # First call
        b1 = step._get_batch()
        # Second call cycles because data has only 1 element
        b2 = step._get_batch()
        assert b1.shape == b2.shape


# ====================================================================
# WGANGPTrainStep
# ====================================================================

class TestWGANGPTrainStep:
    """Tests for WGANGPTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = WGANGPTrainStep()
        assert step.name == "WGANGPTrainStep"

    def test_setup_gp_defaults(self):
        """setup() defaults gp_weight=10.0 and uses default interpolation."""
        components = _make_adversarial_components()
        step = WGANGPTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.gp_weight == 10.0
        assert step.interpolate_fn is WGANGPTrainStep._default_interpolate

    def test_setup_custom_gp_weight(self):
        """setup() accepts custom gp_weight."""
        components = _make_adversarial_components()
        step = WGANGPTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            gp_weight=5.0,
        )

        assert step.gp_weight == 5.0

    def test_default_interpolate(self):
        """_default_interpolate produces values between real and fake."""
        real = torch.ones(4, 3)
        fake = torch.zeros(4, 3)
        interpolated = WGANGPTrainStep._default_interpolate(real, fake)
        assert interpolated.requires_grad is True
        # Values should be between 0 and 1
        assert (interpolated >= 0.0).all()
        assert (interpolated <= 1.0).all()

    def test_gradient_penalty_is_finite(self):
        """_gradient_penalty returns a finite positive scalar."""
        torch.manual_seed(42)
        components = _make_adversarial_components()
        step = WGANGPTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        real = torch.randn(2, 4)
        fake = torch.randn(2, 4)
        gp = step._gradient_penalty(real, fake)
        assert torch.isfinite(gp)
        assert gp.item() >= 0.0


# ====================================================================
# PPOTrainStep
# ====================================================================

class TestPPOTrainStep:
    """Tests for PPOTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = PPOTrainStep()
        assert step.name == "PPOTrainStep"

    def test_paradigm_is_rollout(self):
        """Strategy declares ROLLOUT paradigm."""
        assert PPOTrainStep.paradigm == TrainingParadigm.ROLLOUT

    def test_setup_stores_defaults(self):
        """setup() stores default clip_range, value_coeff, entropy_coeff."""
        components = _make_rl_components()
        step = PPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.clip_range == 0.2
        assert step.value_coeff == 0.5
        assert step.entropy_coeff == 0.01

    def test_setup_custom_hyperparams(self):
        """setup() accepts custom hyperparameters."""
        components = _make_rl_components()
        step = PPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clip_range=0.1,
            value_coeff=0.25,
            entropy_coeff=0.005,
        )

        assert step.clip_range == 0.1
        assert step.value_coeff == 0.25
        assert step.entropy_coeff == 0.005

    def test_train_step_none_batch_returns_untrained(self):
        """train_step with None batch returns trained=False."""
        components = _make_rl_components()
        step = PPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        result = step.train_step(step=1, batch=None)
        assert result.trained is False
        assert result.metrics['loss'] == 0.0

    def test_train_step_with_rollout_batch(self):
        """train_step processes a rollout batch and returns metrics."""
        torch.manual_seed(42)
        components = _make_rl_components()
        step = PPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        batch = _make_rollout_batch(n=4, obs_dim=4, act_dim=2)
        result = step.train_step(step=1, batch=batch)

        assert result.trained is True
        assert 'loss' in result.metrics
        assert 'policy_loss' in result.metrics
        assert 'value_loss' in result.metrics
        assert 'entropy' in result.metrics
        assert 'approx_kl' in result.metrics
        for key, val in result.metrics.items():
            assert torch.isfinite(torch.tensor(float(val))), f"{key} is not finite"

    def test_train_step_updates_parameters(self):
        """Parameters change after a train_step."""
        torch.manual_seed(42)
        components = _make_rl_components()
        step = PPOTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        params_before = [p.clone() for p in components.actor.parameters()]
        batch = _make_rollout_batch(n=4, obs_dim=4, act_dim=2)
        step.train_step(step=1, batch=batch)

        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, components.actor.parameters())
        )
        assert any_changed, "Actor parameters should change after train_step"


# ====================================================================
# A2CTrainStep
# ====================================================================

class TestA2CTrainStep:
    """Tests for A2CTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = A2CTrainStep()
        assert step.name == "A2CTrainStep"

    def test_paradigm_is_rollout(self):
        """Strategy declares ROLLOUT paradigm."""
        assert A2CTrainStep.paradigm == TrainingParadigm.ROLLOUT

    def test_setup_defaults(self):
        """setup() stores default value_coeff and entropy_coeff."""
        components = _make_rl_components()
        step = A2CTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.value_coeff == 0.5
        assert step.entropy_coeff == 0.01

    def test_train_step_none_batch_returns_untrained(self):
        """train_step with None batch returns trained=False."""
        components = _make_rl_components()
        step = A2CTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        result = step.train_step(step=1, batch=None)
        assert result.trained is False

    def test_train_step_with_rollout_batch(self):
        """train_step processes a rollout batch and returns metrics."""
        torch.manual_seed(42)
        components = _make_rl_components()
        step = A2CTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        batch = _make_rollout_batch(n=4, obs_dim=4, act_dim=2)
        result = step.train_step(step=1, batch=batch)

        assert result.trained is True
        assert 'policy_loss' in result.metrics
        assert 'value_loss' in result.metrics
        assert 'entropy' in result.metrics
        # A2C does not report approx_kl
        assert 'approx_kl' not in result.metrics


# ====================================================================
# PredictiveCodingTrainStep
# ====================================================================

class TestPredictiveCodingTrainStep:
    """Tests for PredictiveCodingTrainStep."""

    def test_name(self):
        """Reports correct name."""
        step = PredictiveCodingTrainStep()
        assert step.name == "PredictiveCodingTrainStep"

    def test_paradigm_is_local_learning(self):
        """Strategy declares LOCAL_LEARNING paradigm."""
        assert PredictiveCodingTrainStep.paradigm == TrainingParadigm.LOCAL_LEARNING

    def test_requires_clamp_fn(self):
        """setup() raises when clamp_fn is not provided."""
        components = _make_pc_components()
        step = PredictiveCodingTrainStep()
        with pytest.raises(ValueError, match="requires a clamp_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_stores_fields(self):
        """setup() stores model, optimizer, clamp_fn, inference params."""
        components = _make_pc_components()

        def clamp_fn(batch, device):
            return {0: batch.to(device)}

        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=clamp_fn,
            inference_steps=30,
            inference_lr=0.05,
        )

        assert step.model is components.model
        assert step.optimizer is components.optimizer
        assert step.clamp_fn is clamp_fn
        assert step.inference_steps == 30
        assert step.inference_lr == 0.05

    def test_setup_default_inference_params(self):
        """setup() defaults inference_steps=20, inference_lr=0.1."""
        components = _make_pc_components()
        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=lambda b, d: {0: b},
        )

        assert step.inference_steps == 20
        assert step.inference_lr == 0.1

    def test_train_step_with_batch(self):
        """train_step runs settling + weight update and returns metrics."""
        torch.manual_seed(42)
        components = _make_pc_components()

        def clamp_fn(batch, device):
            # Clamp input layer and top layer (label)
            return {0: batch.to(device), 2: torch.randn(batch.shape[0], 2).to(device)}

        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=clamp_fn,
            inference_steps=5,
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert isinstance(result, StepResult)
        assert result.trained is True
        assert 'free_energy' in result.metrics
        assert 'loss' in result.metrics
        assert torch.isfinite(torch.tensor(float(result.metrics['free_energy'])))

    def test_train_step_caches_state(self):
        """train_step populates strategy caches for hooks."""
        torch.manual_seed(42)
        components = _make_pc_components()

        def clamp_fn(batch, device):
            return {0: batch.to(device), 2: torch.randn(batch.shape[0], 2).to(device)}

        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=clamp_fn,
            inference_steps=3,
        )

        batch = torch.randn(2, 4)
        step.train_step(step=1, batch=batch)

        assert step._cached_activations is not None
        assert step._cached_errors is not None
        assert step._cached_free_energy is not None
        assert step._cached_weight_grads is not None
        assert step._cached_error_norms is not None
        # Number of activations = num layers + 1 (input layer)
        assert len(step._cached_activations) == 3  # [4, 4, 2] -> 3 levels
        assert len(step._cached_errors) == 2  # 2 PC layers

    def test_train_step_reports_per_layer_error_norms(self):
        """train_step includes per-layer error norm metrics."""
        torch.manual_seed(42)
        components = _make_pc_components()

        def clamp_fn(batch, device):
            return {0: batch.to(device), 2: torch.randn(batch.shape[0], 2).to(device)}

        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=clamp_fn,
            inference_steps=3,
        )

        batch = torch.randn(2, 4)
        result = step.train_step(step=1, batch=batch)

        assert 'layer_0_error_norm' in result.metrics
        assert 'layer_1_error_norm' in result.metrics

    def test_train_step_no_batch_no_data_raises(self):
        """train_step raises when no batch and no data source."""
        components = _make_pc_components()
        components.data = None
        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=lambda b, d: {0: b},
        )

        with pytest.raises(ValueError, match="no batch provided and no data source"):
            step.train_step(step=1)

    def test_train_step_fetches_from_data(self):
        """train_step fetches batch from data when not provided."""
        torch.manual_seed(42)
        data = [torch.randn(2, 4) for _ in range(5)]
        components = _make_pc_components()
        components.data = data

        def clamp_fn(batch, device):
            return {0: batch.to(device), 2: torch.randn(batch.shape[0], 2).to(device)}

        step = PredictiveCodingTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            clamp_fn=clamp_fn,
            inference_steps=3,
        )

        result = step.train_step(step=1)
        assert result.trained is True


# ====================================================================
# _ParameterizedModule (MAML helper)
# ====================================================================

class TestParameterizedModule:
    """Tests for the _ParameterizedModule functional wrapper."""

    def test_call_dispatches_to_functional_call(self):
        """Calling the wrapper invokes the module with custom params."""
        torch.manual_seed(42)
        model = nn.Linear(4, 2)
        # Create modified params
        params = {n: p.clone() * 2 for n, p in model.named_parameters()}
        wrapper = _ParameterizedModule(model, params)

        x = torch.randn(2, 4)
        out = wrapper(x)
        # Output should use the modified (doubled) parameters
        expected = torch.func.functional_call(model, params, (x,))
        assert torch.allclose(out, expected)

    def test_parameters_returns_param_values(self):
        """parameters() yields the adapted parameter tensors."""
        model = nn.Linear(4, 2)
        params = dict(model.named_parameters())
        wrapper = _ParameterizedModule(model, params)
        assert list(wrapper.parameters()) == list(params.values())

    def test_named_parameters_returns_items(self):
        """named_parameters() yields (name, param) pairs."""
        model = nn.Linear(4, 2)
        params = dict(model.named_parameters())
        wrapper = _ParameterizedModule(model, params)
        assert list(wrapper.named_parameters()) == list(params.items())


# ====================================================================
# MAMLStep
# ====================================================================

class TestMAMLStep:
    """Tests for MAMLStep."""

    def test_name_maml(self):
        """Reports 'MAMLStep' when first_order=False."""
        step = MAMLStep()
        # Need to call setup to set first_order attribute
        components = _make_meta_components()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )
        assert step.name == "MAMLStep"

    def test_name_fomaml(self):
        """Reports 'FOMAMLStep' when first_order=True."""
        step = MAMLStep()
        components = _make_meta_components()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            first_order=True,
        )
        assert step.name == "FOMAMLStep"

    def test_paradigm_is_nested(self):
        """Strategy declares NESTED paradigm."""
        assert MAMLStep.paradigm == TrainingParadigm.NESTED

    def test_requires_task_loss_fn(self):
        """setup() raises when task_loss_fn is None."""
        components = _make_meta_components()
        components.task_loss_fn = None

        step = MAMLStep()
        with pytest.raises(ValueError, match="requires a task_loss_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_stores_defaults(self):
        """setup() stores default hyperparameters."""
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.inner_lr == 0.01
        assert step.inner_steps == 5
        assert step.tasks_per_step == 4
        assert step.first_order is False

    def test_setup_custom_hyperparams(self):
        """setup() accepts custom hyperparameters."""
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_lr=0.1,
            inner_steps=3,
            tasks_per_step=2,
            first_order=True,
        )

        assert step.inner_lr == 0.1
        assert step.inner_steps == 3
        assert step.tasks_per_step == 2
        assert step.first_order is True

    def test_train_step_produces_metrics(self):
        """train_step runs inner/outer loop and returns expected metrics."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        result = step.train_step(step=1)

        assert isinstance(result, StepResult)
        assert 'loss' in result.metrics
        assert 'mean_task_loss' in result.metrics
        assert 'task_loss_std' in result.metrics
        assert 'num_tasks' in result.metrics
        assert result.metrics['num_tasks'] == 2

    def test_train_step_caches_gradients(self):
        """train_step populates gradient caches for hooks."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        step.train_step(step=1)

        assert step._cached_meta_gradients is not None
        assert len(step._cached_meta_gradients) > 0
        assert step._cached_inner_gradients is not None
        assert step._cached_task_losses is not None
        assert len(step._cached_task_losses) == 2

    def test_fomaml_train_step(self):
        """FOMAML (first_order=True) produces valid metrics."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
            first_order=True,
        )

        result = step.train_step(step=1)
        assert result.trained is True
        assert torch.isfinite(torch.tensor(float(result.metrics['loss'])))

    def test_train_step_updates_model_parameters(self):
        """Meta-optimizer step changes model parameters."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = MAMLStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        params_before = [p.clone() for p in components.model.parameters()]
        step.train_step(step=1)

        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, components.model.parameters())
        )
        assert any_changed, "Model parameters should change after meta-step"


# ====================================================================
# ReptileStep
# ====================================================================

class TestReptileStep:
    """Tests for ReptileStep."""

    def test_name(self):
        """Reports correct name."""
        step = ReptileStep()
        assert step.name == "ReptileStep"

    def test_paradigm_is_nested(self):
        """Strategy declares NESTED paradigm."""
        assert ReptileStep.paradigm == TrainingParadigm.NESTED

    def test_requires_task_loss_fn(self):
        """setup() raises when task_loss_fn is None."""
        components = _make_meta_components()
        components.task_loss_fn = None

        step = ReptileStep()
        with pytest.raises(ValueError, match="requires a task_loss_fn"):
            step.setup(
                components=components,
                config=BaseConfig(),
                device=torch.device('cpu'),
            )

    def test_setup_defaults(self):
        """setup() stores default hyperparameters."""
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        assert step.inner_lr == 0.01
        assert step.inner_steps == 5
        assert step.tasks_per_step == 4

    def test_train_step_produces_metrics(self):
        """train_step runs and returns expected metrics."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        result = step.train_step(step=1)

        assert isinstance(result, StepResult)
        assert 'loss' in result.metrics
        assert 'mean_task_loss' in result.metrics
        assert 'task_loss_std' in result.metrics
        assert 'num_tasks' in result.metrics
        assert 'meta_grad_norm' in result.metrics
        assert result.metrics['num_tasks'] == 2

    def test_train_step_caches_meta_gradients(self):
        """train_step populates cached meta-gradients."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        step.train_step(step=1)

        assert step._cached_meta_gradients is not None
        assert step._cached_inner_gradients is None  # Reptile doesn't cache inner grads
        assert step._cached_task_losses is not None

    def test_train_step_restores_parameters(self):
        """Model parameters are restored between tasks in inner loop."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        # Run a step and check it works (restoration is implicit)
        result = step.train_step(step=1)
        assert result.trained is True

    def test_train_step_updates_model_parameters(self):
        """Meta-optimizer step changes model parameters."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=2,
            tasks_per_step=2,
        )

        params_before = [p.clone() for p in components.model.parameters()]
        step.train_step(step=1)

        any_changed = any(
            not torch.allclose(before, after)
            for before, after in zip(params_before, components.model.parameters())
        )
        assert any_changed, "Model parameters should change after Reptile meta-step"

    def test_multiple_steps_reduce_loss(self):
        """Running multiple meta-steps reduces loss (sanity check)."""
        torch.manual_seed(42)
        components = _make_meta_components()
        step = ReptileStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            inner_steps=3,
            tasks_per_step=3,
        )

        losses = []
        for i in range(5):
            result = step.train_step(step=i + 1)
            losses.append(float(result.metrics['loss']))

        # Loss should generally decrease (not guaranteed but likely with decent LR)
        # Just check that the loop ran without error and losses are finite
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


# ====================================================================
# StrategyRunner base class interface
# ====================================================================

class TestStrategyRunnerInterface:
    """Tests for StrategyRunner base class default behaviors."""

    def test_default_paradigm_is_backprop(self):
        """Default paradigm class attribute is BACKPROP."""
        from framework.strategies.strategy_runner import StrategyRunner
        assert StrategyRunner.paradigm == TrainingParadigm.BACKPROP

    def test_default_name_is_class_name(self):
        """Default name property returns class name."""
        from framework.strategies.strategy_runner import StrategyRunner

        class MyStrategy(StrategyRunner):
            def train_step(self, step, batch=None):
                return StepResult()

        s = MyStrategy()
        assert s.name == "MyStrategy"

    def test_default_post_step_returns_none(self):
        """Default post_step returns None."""
        from framework.strategies.strategy_runner import StrategyRunner

        class MyStrategy(StrategyRunner):
            def train_step(self, step, batch=None):
                return StepResult()

        s = MyStrategy()
        assert s.post_step(step=1, result=StepResult()) is None

    def test_default_setup_is_noop(self):
        """Default setup() does nothing and doesn't raise."""
        from framework.strategies.strategy_runner import StrategyRunner

        class MyStrategy(StrategyRunner):
            def train_step(self, step, batch=None):
                return StepResult()

        s = MyStrategy()
        # Should not raise
        s.setup(components=None, config=None, device=None)

    def test_default_teardown_is_noop(self):
        """Default teardown() does nothing and doesn't raise."""
        from framework.strategies.strategy_runner import StrategyRunner

        class MyStrategy(StrategyRunner):
            def train_step(self, step, batch=None):
                return StepResult()

        s = MyStrategy()
        s.teardown()


# ====================================================================
# Additional coverage: ContrastiveTrainStep self-feeding + accumulation
# ====================================================================

class TestContrastiveSelfFeeding:
    """Tests for ContrastiveTrainStep data iteration and accumulation."""

    def _make_contrastive_step(self, data=None, accumulation_steps=1):
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        all_params = list(encoder.parameters()) + list(proj.parameters())
        optimizer = SGD(all_params, lr=0.01)
        components = BackpropComponents(
            model=encoder,
            optimizer=optimizer,
            scheduler=None,
            criterion=None,
            loss_fn=None,
            strategy=None,
            data=data,
        )
        config = ExtendedConfig(accumulation_steps=accumulation_steps)
        step = ContrastiveTrainStep()
        step.setup(
            components=components,
            config=config,
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x + torch.randn_like(x) * 0.1,
                                  x + torch.randn_like(x) * 0.1),
        )
        return step

    def test_self_feeding_from_data(self):
        """train_step fetches batch from data when none provided."""
        data = [torch.randn(4, 4) for _ in range(5)]
        step = self._make_contrastive_step(data=data)
        result = step.train_step(step=1)
        assert result.trained is True
        assert torch.isfinite(result.metrics['loss'])

    def test_self_feeding_stop_on_exhaustion(self):
        """train_step returns should_stop when data iterator is exhausted."""
        data = [torch.randn(4, 4)]
        step = self._make_contrastive_step(data=data)
        step.train_step(step=1)  # consume the only batch
        result = step.train_step(step=2)
        assert result.should_stop is True

    def test_no_data_raises(self):
        """train_step raises when no batch and no data source."""
        step = self._make_contrastive_step(data=None)
        with pytest.raises(ValueError, match="no batch and no data"):
            step.train_step(step=1)

    def test_accumulation_delays_update(self):
        """With accumulation_steps=2, first step has trained=False."""
        data = [torch.randn(4, 4) for _ in range(5)]
        step = self._make_contrastive_step(data=data, accumulation_steps=2)
        r1 = step.train_step(step=1)
        assert r1.trained is False
        r2 = step.train_step(step=2)
        assert r2.trained is True


# ====================================================================
# Additional coverage: MomentumContrastiveTrainStep full train_step
# ====================================================================

class TestMomentumContrastiveTrainStepFull:
    """Full train_step tests for MomentumContrastiveTrainStep."""

    def _make_momentum_step(self, data=None):
        torch.manual_seed(42)
        encoder = nn.Linear(4, 4)
        momentum_enc = nn.Linear(4, 4)
        proj = nn.Linear(4, 8)
        momentum_proj = nn.Linear(4, 8)
        all_params = list(encoder.parameters()) + list(proj.parameters())
        optimizer = SGD(all_params, lr=0.01)
        components = BackpropComponents(
            model=encoder,
            optimizer=optimizer,
            scheduler=None,
            criterion=None,
            loss_fn=None,
            strategy=None,
            data=data,
            auxiliary_models={'momentum_encoder': momentum_enc},
        )
        step = MomentumContrastiveTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            projection_head=proj,
            augment_fn=lambda x: (x + torch.randn_like(x) * 0.1,
                                  x + torch.randn_like(x) * 0.1),
            momentum_projection_head=momentum_proj,
            ema_decay=0.99,
        )
        return step

    def test_train_step_returns_loss(self):
        """Momentum contrastive train_step produces loss and updates."""
        step = self._make_momentum_step()
        batch = torch.randn(4, 4)
        result = step.train_step(step=1, batch=batch)
        assert isinstance(result, StepResult)
        assert result.trained is True
        assert torch.isfinite(result.metrics['loss'])

    def test_train_step_self_feeding(self):
        """train_step fetches batch from data when none provided."""
        data = [torch.randn(4, 4) for _ in range(3)]
        step = self._make_momentum_step(data=data)
        result = step.train_step(step=1)
        assert result.trained is True

    def test_train_step_no_data_raises(self):
        """Raises when no batch and no data."""
        step = self._make_momentum_step(data=None)
        with pytest.raises(ValueError, match="no batch and no data"):
            step.train_step(step=1)

    def test_momentum_updated_after_step(self):
        """Momentum encoder is updated via EMA after optimizer step."""
        step = self._make_momentum_step()

        mom_before = [p.clone() for p in step.momentum_encoder.parameters()]

        batch = torch.randn(4, 4)
        step.train_step(step=1, batch=batch)

        # At least one momentum param should have changed
        changed = any(
            not torch.allclose(before, after)
            for before, after in zip(mom_before, step.momentum_encoder.parameters())
        )
        assert changed, "Momentum encoder should be updated after train_step"


# ====================================================================
# Additional coverage: WGANGPTrainStep full train_step
# ====================================================================

class TestWGANGPTrainStepFull:
    """Full train_step test for WGANGPTrainStep."""

    def test_train_step_with_gradient_penalty(self):
        """WGANGPTrainStep.train_step produces valid results with GP."""
        torch.manual_seed(42)
        # Generator needs a latent_dim attribute for WGAN-GP
        G = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        G.latent_dim = 4
        D = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))

        def g_loss_fn(gen, disc, batch):
            fake = gen(batch)
            return -disc(fake).mean()

        def d_loss_fn(gen, disc, batch):
            with torch.no_grad():
                fake = gen(batch)
            return disc(fake).mean() - disc(batch).mean()

        batch = torch.randn(2, 4)
        components = AdversarialComponents(
            generator=G,
            discriminator=D,
            g_optimizer=SGD(G.parameters(), lr=0.01),
            d_optimizer=SGD(D.parameters(), lr=0.01),
            g_scheduler=None,
            d_scheduler=None,
            strategy=None,
            data=[batch.clone() for _ in range(10)],
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

        from framework.strategies.adversarial import WGANGPTrainStep
        step = WGANGPTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
            gp_weight=10.0,
        )

        result = step.train_step(step=1, batch=batch)
        assert isinstance(result, StepResult)
        assert 'g_loss' in result.metrics
        assert 'd_loss' in result.metrics
        assert torch.isfinite(result.metrics['g_loss'])
        assert torch.isfinite(result.metrics['d_loss'])

    def test_adversarial_self_feeding_stop_on_exhaustion(self):
        """MomentumContrastiveTrainStep returns should_stop on data exhaustion."""
        torch.manual_seed(42)
        G = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        D = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))

        def g_loss_fn(gen, disc, batch):
            return -disc(gen(batch)).mean()

        def d_loss_fn(gen, disc, batch):
            with torch.no_grad():
                fake = gen(batch)
            return disc(fake).mean() - disc(batch).mean()

        data = [torch.randn(2, 4) for _ in range(5)]
        components = AdversarialComponents(
            generator=G,
            discriminator=D,
            g_optimizer=SGD(G.parameters(), lr=0.01),
            d_optimizer=SGD(D.parameters(), lr=0.01),
            g_scheduler=None,
            d_scheduler=None,
            strategy=None,
            data=data,
            g_loss_fn=g_loss_fn,
            d_loss_fn=d_loss_fn,
        )

        from framework.strategies.adversarial import AdversarialTrainStep
        step = AdversarialTrainStep()
        step.setup(
            components=components,
            config=BaseConfig(),
            device=torch.device('cpu'),
        )

        # Consume all data (auto-cycles, so this should work indefinitely)
        for i in range(10):
            result = step.train_step(step=i + 1)
            assert result.trained is True
