"""Tests for framework/trainers/adversarial_trainer.py — adversarial (GAN) training loop."""

import pytest
import torch

from framework.trainers.adversarial_trainer import AdversarialStepTrainer
from framework.trainers.base import TrainResult
from framework.hooks.hook_point import HookPoint
from framework.hooks.training_hook import TrainingHook
from framework.hooks.manager import HookManager


# ---- Mock hooks ----

class StepCountingHook(TrainingHook):
    """Hook that counts how many times it fires at each point."""
    name = "adv_step_counter"
    hook_points = {HookPoint.POST_STEP, HookPoint.SNAPSHOT}

    def __init__(self):
        self.fire_counts = {hp: 0 for hp in HookPoint}

    def compute(self, ctx, **state):
        self.fire_counts[ctx.hook_point] += 1
        return {"count": self.fire_counts[ctx.hook_point]}


# ---- Basic loop execution ----

class TestAdversarialTrainerBasicLoop:

    def test_runs_to_completion(self, make_runner):
        """5-step adversarial training runs and returns results dict."""
        runner = make_runner(steps=5, component_type='adversarial')
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        results = trainer.train()
        assert 'test_strategy' in results

    def test_train_result_fields(self, make_runner):
        """TrainResult has correct final_step and early_stopped=False."""
        runner = make_runner(steps=5, component_type='adversarial')
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert isinstance(result, TrainResult)
        assert result.final_step_or_epoch == 5
        assert result.early_stopped is False

    def test_generator_params_change(self, make_runner):
        """Generator parameters change during training."""
        runner = make_runner(steps=3, component_type='adversarial')
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        gen = loop_state.components.generator
        params_before = [p.clone() for p in gen.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        for p_before, p_after in zip(params_before, gen.parameters()):
            assert not torch.allclose(p_before, p_after)

    def test_discriminator_params_change(self, make_runner):
        """Discriminator parameters change during training."""
        runner = make_runner(steps=5, component_type='adversarial')
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        disc = loop_state.components.discriminator
        params_before = [p.clone() for p in disc.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, disc.parameters())
        )
        assert changed, "Discriminator parameters should change after training"


# ---- Hook firing ----

class TestAdversarialTrainerHooks:

    def test_post_step_hooks_fire(self, make_runner):
        """POST_STEP hooks fire every step."""
        runner = make_runner(steps=3, component_type='adversarial')
        hook = StepCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        assert hook.fire_counts[HookPoint.POST_STEP] == 3

    def test_snapshot_hooks_fire_at_intervals(self, make_runner):
        """SNAPSHOT hooks fire at snapshot_every intervals."""
        runner = make_runner(steps=4, snapshot_every=2, component_type='adversarial')
        hook = StepCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Steps 2 and 4 are snapshot points
        assert hook.fire_counts[HookPoint.SNAPSHOT] == 2


# ---- Trajectory ----

class TestAdversarialTrainerTrajectory:

    def test_trajectory_records_gan_losses(self, make_runner):
        """record_trajectory=True stores g_loss and d_loss in trajectory entries."""
        runner = make_runner(
            steps=4, snapshot_every=2,
            record_trajectory=True, component_type='adversarial',
        )
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        assert loop_state.trajectory is not None
        trainer._training_loop('test_strategy', loop_state)
        # Snapshots at steps 2 and 4
        assert len(loop_state.trajectory) == 2
        entry = loop_state.trajectory[0]
        assert 'g_loss' in entry
        assert 'd_loss' in entry
        assert 'params' in entry
        assert entry['step'] == 2


# ---- Early stopping ----

class TestAdversarialTrainerEarlyStopping:

    def test_runner_should_stop(self, make_runner):
        """runner.should_stop() triggers exit at the designated step."""
        runner = make_runner(
            steps=10, eval_every=2, stop_at=2,
            component_type='adversarial',
        )
        trainer = AdversarialStepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch == 2
        assert result.early_stopped is True
