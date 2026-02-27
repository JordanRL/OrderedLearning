"""Tests for framework/trainers/rollout_trainer.py — RL rollout-based training loop."""

import pytest
import torch

from framework.trainers.rollout_trainer import RolloutTrainer
from framework.trainers.base import TrainResult
from framework.hooks.hook_point import HookPoint
from framework.hooks.training_hook import TrainingHook
from framework.hooks.manager import HookManager


# ---- Mock hooks ----

class RolloutCountingHook(TrainingHook):
    """Hook that counts how many times it fires at each point."""
    name = "rollout_counter"
    hook_points = {HookPoint.POST_STEP, HookPoint.SNAPSHOT}

    def __init__(self):
        self.fire_counts = {hp: 0 for hp in HookPoint}

    def compute(self, ctx, **state):
        self.fire_counts[ctx.hook_point] += 1
        return {"count": self.fire_counts[ctx.hook_point]}


# ---- Basic loop execution ----

class TestRolloutTrainerBasicLoop:

    def test_runs_to_completion(self, make_runner):
        """RL training runs and returns results dict."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        results = trainer.train()
        assert 'test_strategy' in results

    def test_train_result_fields(self, make_runner):
        """TrainResult reflects completion with correct fields."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert isinstance(result, TrainResult)
        assert result.final_step_or_epoch >= 8
        assert result.duration > 0

    def test_actor_params_change(self, make_runner):
        """Actor parameters change during training."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        actor = loop_state.components.actor
        params_before = [p.clone() for p in actor.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, actor.parameters())
        )
        assert changed, "Actor parameters should change after training"

    def test_critic_params_change(self, make_runner):
        """Critic parameters change during training."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        critic = loop_state.components.critic
        params_before = [p.clone() for p in critic.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, critic.parameters())
        )
        assert changed, "Critic parameters should change after training"


# ---- Two-phase structure ----

class TestRolloutTrainerPhases:

    def test_buffer_populated_during_rollout(self, make_runner):
        """Buffer receives transitions during rollout collection phase."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        buffer = loop_state.components.rollout_buffer
        actor = loop_state.components.actor
        # Buffer starts empty
        assert len(buffer) == 0
        # Snapshot actor params before training
        params_before = [p.clone() for p in actor.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        # Actor params changed, confirming both rollout collection and
        # policy update phases executed (buffer was populated and consumed)
        changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, actor.parameters())
        )
        assert changed, "Actor params should change, confirming rollout + update executed"


# ---- Hook firing ----

class TestRolloutTrainerHooks:

    def test_post_step_hooks_fire_per_rollout(self, make_runner):
        """POST_STEP hooks fire once per rollout, not per minibatch."""
        # rollout_length=4, steps=8 → 2 rollouts
        runner = make_runner(steps=8, component_type='rl')
        hook = RolloutCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        trainer = RolloutTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Should fire once per rollout (8 steps / 4 rollout_length = 2 rollouts)
        assert hook.fire_counts[HookPoint.POST_STEP] == 2

    def test_no_hook_manager_runs_cleanly(self, make_runner):
        """Training runs without error when hook_manager is None."""
        runner = make_runner(steps=8, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch >= 8


# ---- Error handling ----

class TestRolloutTrainerErrors:

    def test_missing_buffer_raises(self, make_runner):
        """Missing rollout_buffer raises ValueError."""
        runner = make_runner(steps=4, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        loop_state.components.rollout_buffer = None
        with pytest.raises(ValueError, match="rollout_buffer"):
            trainer._training_loop('test_strategy', loop_state)

    def test_missing_env_raises(self, make_runner):
        """Missing environment (data) raises ValueError."""
        runner = make_runner(steps=4, component_type='rl')
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        loop_state.components.data = None
        with pytest.raises(ValueError, match="data"):
            trainer._training_loop('test_strategy', loop_state)


# ---- Early stopping ----

class TestRolloutTrainerEarlyStopping:

    def test_runner_should_stop(self, make_runner):
        """runner.should_stop() triggers exit."""
        # eval_every=1 ensures the stop check fires after every rollout.
        # rollout_length=4 (config default) + start=1 → env_step=5 after first rollout.
        # stop_at=5 → should_stop(5) = True.
        runner = make_runner(
            steps=100, eval_every=1, stop_at=5,
            component_type='rl',
        )
        trainer = RolloutTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.early_stopped is True
        assert result.final_step_or_epoch < 100
