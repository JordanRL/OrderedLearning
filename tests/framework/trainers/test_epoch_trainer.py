"""Tests for framework/trainers/epoch_trainer.py — epoch-based training loop."""

import pytest
import torch

from framework.trainers.epoch_trainer import EpochTrainer
from framework.trainers.base import TrainResult
from framework.hooks.hook_point import HookPoint
from framework.hooks.training_hook import TrainingHook
from framework.hooks.manager import HookManager


# ---- Mock hooks ----

class EpochCountingHook(TrainingHook):
    """Hook that counts fires per hook point."""
    name = "epoch_counter"
    hook_points = {HookPoint.POST_EPOCH, HookPoint.POST_STEP, HookPoint.SNAPSHOT}

    def __init__(self):
        self.fire_counts = {hp: 0 for hp in HookPoint}

    def compute(self, ctx, **state):
        self.fire_counts[ctx.hook_point] += 1
        return {"count": self.fire_counts[ctx.hook_point]}


# ---- Basic loop execution ----

class TestEpochTrainerBasicLoop:

    def test_runs_to_completion(self, make_runner):
        """3-epoch training runs and returns results."""
        runner = make_runner(epochs=3)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        results = trainer.train()
        assert 'test_strategy' in results

    def test_train_result_fields(self, make_runner):
        """TrainResult has correct final_epoch and global_step."""
        runner = make_runner(epochs=3)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert isinstance(result, TrainResult)
        # Epochs are 0-indexed, so last epoch is 2
        assert result.final_step_or_epoch == 2
        assert result.early_stopped is False
        # 3 epochs x 2 batches = 6 global steps
        assert result.global_step == 6

    def test_parameters_change(self, make_runner):
        """Model parameters actually change during training."""
        runner = make_runner(epochs=2)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        model = loop_state.components.get_primary_model()
        params_before = [p.clone() for p in model.parameters()]
        trainer._training_loop('test_strategy', loop_state)
        for p_before, p_after in zip(params_before, model.parameters()):
            assert not torch.allclose(p_before, p_after)


# ---- Epoch semantics ----

class TestEpochTrainerSemantics:

    def test_scheduler_steps_per_epoch(self, make_runner):
        """Scheduler steps once per epoch, not per batch."""
        runner = make_runner(epochs=3)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        lr_before = loop_state.components.get_lr()
        trainer._training_loop('test_strategy', loop_state)
        lr_after = loop_state.components.get_lr()
        # StepLR(gamma=0.9) after 3 steps: 0.1 * 0.9^3 = 0.0729
        assert lr_after < lr_before
        assert lr_after == pytest.approx(0.1 * 0.9**3, rel=1e-4)

    def test_global_step_increments_per_batch(self, make_runner):
        """global_step counts total batches across all epochs."""
        runner = make_runner(epochs=2)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        # 2 epochs x 2 batches per epoch = 4
        assert result.global_step == 4


# ---- Hook firing ----

class TestEpochTrainerHooks:

    def test_post_epoch_fires_every_epoch(self, make_runner):
        """POST_EPOCH fires once per epoch."""
        runner = make_runner(epochs=3)
        hook = EpochCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None, loop_type='epoch')
        trainer = EpochTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        assert hook.fire_counts[HookPoint.POST_EPOCH] == 3

    def test_post_step_fires_every_batch(self, make_runner):
        """POST_STEP fires once per batch (2 batches x 3 epochs = 6)."""
        runner = make_runner(epochs=3)
        hook = EpochCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None, loop_type='epoch')
        trainer = EpochTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        assert hook.fire_counts[HookPoint.POST_STEP] == 6

    def test_snapshot_fires_at_intervals(self, make_runner):
        """SNAPSHOT fires at snapshot_every epoch intervals."""
        runner = make_runner(epochs=4, snapshot_every=2)
        hook = EpochCountingHook()
        hm = HookManager(hooks=[hook], step_metrics_log=None, loop_type='epoch')
        trainer = EpochTrainer(runner=runner, hook_manager=hm)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Epochs 0 and 2 are snapshot points (0 % 2 == 0, 2 % 2 == 0)
        assert hook.fire_counts[HookPoint.SNAPSHOT] == 2

    def test_no_hook_manager_runs_cleanly(self, make_runner):
        """Training runs without error when hook_manager is None."""
        runner = make_runner(epochs=2)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch == 1  # 0-indexed, 2 epochs -> last is 1


# ---- Evaluation ----

class TestEpochTrainerEvaluation:

    def test_eval_called_at_intervals(self, make_runner):
        """With eval_every=2 and 4 epochs, evaluate is called at epochs 0 and 2."""
        runner = make_runner(epochs=4, eval_every=2, snapshot_every=2)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        # Periodic evals at epochs 0 and 2 (eval_every=2)
        assert 0 in runner._evaluate_calls
        assert 2 in runner._evaluate_calls


# ---- Early stopping ----

class TestEpochTrainerEarlyStopping:

    def test_runner_should_stop(self, make_runner):
        """runner.should_stop() at epoch 1 causes early exit."""
        runner = make_runner(epochs=10, stop_at=1)
        trainer = EpochTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        assert result.final_step_or_epoch <= 1
        assert result.early_stopped is True
