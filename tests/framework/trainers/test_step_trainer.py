"""Tests for framework/trainers/step_trainer.py — step-based training loop."""

import pytest
import torch

from framework.trainers.step_trainer import StepTrainer
from framework.trainers.base import TrainResult
from framework.hooks.hook_point import HookPoint
from framework.hooks.training_hook import TrainingHook
from framework.hooks.manager import HookManager


# ---- Mock hooks for hook-firing tests ----

class StepCountingHook(TrainingHook):
    """Hook that counts how many times it fires at each point."""
    name = "step_counter"
    hook_points = {HookPoint.POST_STEP, HookPoint.SNAPSHOT}

    def __init__(self):
        self.fire_counts = {hp: 0 for hp in HookPoint}

    def compute(self, ctx, **state):
        self.fire_counts[ctx.hook_point] += 1
        return {"count": self.fire_counts[ctx.hook_point]}


# ---- Basic loop execution ----

class TestStepTrainerBasicLoop:

    def test_runs_to_completion(self, make_runner):
        """5-step training runs and returns TrainResult."""
        runner = make_runner(steps=5)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        results = trainer.train()
        assert 'test_strategy' in results

    def test_train_result_fields(self, make_runner):
        """TrainResult has correct final_step and early_stopped=False."""
        runner = make_runner(steps=5)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert isinstance(result, TrainResult)
        assert result.final_step_or_epoch == 5
        assert result.early_stopped is False

    def test_parameters_change(self, make_runner):
        """Model parameters actually change during training."""
        runner = make_runner(steps=3)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        model = loop_state.components.get_primary_model()
        params_before = [p.clone() for p in model.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        for p_before, p_after in zip(params_before, model.parameters()):
            assert not torch.allclose(p_before, p_after)

    def test_scheduler_steps(self, make_runner):
        """LR decreases over steps (StepLR with gamma=0.9)."""
        runner = make_runner(steps=3)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        lr_before = loop_state.components.get_lr()
        trainer._training_loop('test_strategy', loop_state)
        lr_after = loop_state.components.get_lr()
        assert lr_after < lr_before


# ---- Evaluation scheduling ----

class TestStepTrainerEvaluation:

    def test_eval_called_at_intervals(self, make_runner):
        """With eval_every=2 and 5 steps, evaluate called at steps 0, 2, 4."""
        runner = make_runner(steps=5, eval_every=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Step 0 (init eval) + steps 2 and 4 (periodic)
        assert 0 in runner._evaluate_calls
        assert 2 in runner._evaluate_calls
        assert 4 in runner._evaluate_calls

    def test_init_eval_runs_when_not_resuming(self, make_runner):
        """Initial evaluation at step 0 runs when not resuming."""
        runner = make_runner(steps=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        assert loop_state.is_resuming is False
        trainer._training_loop('test_strategy', loop_state)
        assert 0 in runner._evaluate_calls


# ---- Hook firing ----

class TestStepTrainerHooks:

    def test_post_step_hooks_fire(self, make_runner):
        """POST_STEP hooks fire every step when registered."""
        runner = make_runner(steps=3)
        hook = StepCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        trainer = StepTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        assert hook.fire_counts[HookPoint.POST_STEP] == 3

    def test_snapshot_hooks_fire_at_intervals(self, make_runner):
        """SNAPSHOT hooks fire at snapshot_every intervals."""
        runner = make_runner(steps=4, snapshot_every=2)
        hook = StepCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None)
        trainer = StepTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Steps 2 and 4 are snapshot points (2 % 2 == 0, 4 % 2 == 0)
        assert hook.fire_counts[HookPoint.SNAPSHOT] == 2

    def test_no_hook_manager_runs_cleanly(self, make_runner):
        """Training runs without error when hook_manager is None."""
        runner = make_runner(steps=3)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch == 3


# ---- Early stopping ----

class TestStepTrainerEarlyStopping:

    def test_runner_should_stop(self, make_runner):
        """runner.should_stop() at step 2 causes exit after eval at step 2."""
        runner = make_runner(steps=10, eval_every=2, stop_at=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch == 2
        assert result.early_stopped is True


# ---- Trajectory ----

class TestStepTrainerTrajectory:

    def test_trajectory_accumulates(self, make_runner):
        """record_trajectory=True populates trajectory at snapshot intervals."""
        runner = make_runner(steps=4, snapshot_every=2, record_trajectory=True)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        assert loop_state.trajectory is not None
        trainer._training_loop('test_strategy', loop_state)
        # Snapshots at steps 2 and 4
        assert len(loop_state.trajectory) == 2
        assert loop_state.trajectory[0]['step'] == 2
        assert 'params' in loop_state.trajectory[0]
