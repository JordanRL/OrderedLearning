"""Tests for framework/trainers/base.py — Trainer ABC orchestration."""

import os
import pytest
import torch

from framework.trainers.step_trainer import StepTrainer
from framework.trainers.base import TrainResult
from framework.capabilities import TrainingParadigm, GradientAvailability, ModelCapability
from framework.hooks.manager import HookManager
from framework.checkpoints.resume import ResumeInfo


# ---- Strategy iteration ----

class TestTrainerStrategyIteration:

    def test_two_strategies_both_run(self, make_runner):
        """Multiple strategies are each executed."""
        runner = make_runner(strategies=['strat_a', 'strat_b'], steps=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        results = trainer.train()
        assert 'strat_a' in results
        assert 'strat_b' in results

    def test_resume_skips_completed_strategies(self, make_runner):
        """Completed strategies from resume are skipped."""
        runner = make_runner(strategies=['strat_a', 'strat_b'], steps=2)
        resume = ResumeInfo(
            checkpoint_path=None,
            start_step_or_epoch=0,
            completed_strategies=['strat_a'],
            config_path='',
        )
        trainer = StepTrainer(runner=runner, hook_manager=None, resume=resume)
        results = trainer.train()
        assert 'strat_a' not in results
        assert 'strat_b' in results


# ---- Setup ----

class TestTrainerSetup:

    def test_setup_creates_output_dir(self, make_runner):
        """_setup_strategy creates the experiment output directory."""
        runner = make_runner()
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        assert os.path.isdir(loop_state.experiment_dir)

    def test_setup_wires_strategy(self, make_runner):
        """After setup, components.strategy has been set up."""
        runner = make_runner()
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        # SimpleTrainStep.setup() stores self.model — verify it's set
        assert hasattr(loop_state.components.strategy, 'model')
        assert loop_state.components.strategy.model is not None

    def test_setup_with_hook_manager_resets(self, make_runner):
        """_setup_strategy calls hook_manager.reset_all() and set_run_context()."""
        runner = make_runner()
        hm = HookManager(hooks=[], step_metrics_log=None)
        # Advance global step to verify reset clears it
        hm.advance_step()
        assert hm.global_step == 0

        trainer = StepTrainer(runner=runner, hook_manager=hm)
        trainer._setup_strategy('test_strategy')
        # reset_all() should have cleared global_step back to -1
        assert hm.global_step == -1


# ---- Finalization ----

class TestTrainerFinalization:

    def test_finalize_saves_checkpoint(self, make_runner):
        """_finalize_strategy saves a completion checkpoint."""
        runner = make_runner(steps=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        trainer._finalize_strategy('test_strategy', loop_state, result)
        # Check checkpoint file was created
        ckpt_dir = os.path.join(loop_state.experiment_dir, 'checkpoints')
        assert os.path.isdir(ckpt_dir)
        ckpts = os.listdir(ckpt_dir)
        assert len(ckpts) > 0

    def test_finalize_calls_final_eval(self, make_runner):
        """_finalize_strategy triggers a final evaluation."""
        runner = make_runner(steps=2)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        result = trainer._training_loop('test_strategy', loop_state)
        runner._evaluate_calls.clear()
        trainer._finalize_strategy('test_strategy', loop_state, result)
        # Final eval should have been called
        assert len(runner._evaluate_calls) > 0

    def test_resume_consumed_after_first_strategy(self, make_runner):
        """Resume info is consumed (set to None) after first strategy."""
        runner = make_runner(strategies=['s1', 's2'], steps=2)
        resume = ResumeInfo(
            checkpoint_path=None, start_step_or_epoch=0,
            completed_strategies=[], config_path='',
        )
        trainer = StepTrainer(runner=runner, hook_manager=None, resume=resume)
        assert trainer._resume is not None
        trainer.train()
        assert trainer._resume is None


# ---- Capabilities ----

class TestTrainerCapabilities:

    def test_step_trainer_capabilities(self, make_runner):
        """StepTrainer declares BACKPROP paradigm with GLOBAL_GRADIENTS."""
        runner = make_runner()
        trainer = StepTrainer(runner=runner, hook_manager=None)
        caps = trainer.get_capabilities()
        assert caps.paradigm == TrainingParadigm.BACKPROP
        assert caps.gradient_availability == GradientAvailability.GLOBAL_GRADIENTS
        assert caps.model_capabilities == ModelCapability.PARAMETERS


# ---- Periodic checkpoint ----

class TestTrainerPeriodicCheckpoint:

    def test_checkpoint_saved_at_interval(self, make_runner):
        """save_checkpoints=True + checkpoint_every=2 creates checkpoints."""
        runner = make_runner(steps=4, checkpoint_every=2, save_checkpoints=True)
        trainer = StepTrainer(runner=runner, hook_manager=None)
        loop_state = trainer._setup_strategy('test_strategy')
        trainer._training_loop('test_strategy', loop_state)
        ckpt_dir = os.path.join(loop_state.experiment_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_')]
            # Checkpoints at steps 2 and 4
            assert len(ckpts) == 2
