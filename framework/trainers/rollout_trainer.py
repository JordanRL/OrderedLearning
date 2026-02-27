"""Rollout-based RL trainer.

Outer loop: collect rollouts in the environment.
Inner loop: K epochs of policy updates per rollout.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..hooks import HookPoint
from ..capabilities import TrainingParadigm, GradientAvailability
from ..contexts import RunContext, EvalMetrics, BatchState
from .. import display
from .base import Trainer, LoopState, TrainResult


class RolloutTrainer(Trainer):
    """RL training with rollout collection and policy updates.

    Training loop structure:
    1. Collect rollouts: run policy in environment for N steps
    2. Compute advantages (GAE)
    3. Run K epochs of minibatch updates on the rollout data
    4. Repeat

    The 'total' from the runner is total environment steps (rollout iterations).
    """

    paradigm = TrainingParadigm.ROLLOUT
    gradient_availability = GradientAvailability.POLICY_GRADIENTS
    loop_type = 'step'
    _counter_label = "step"
    _default_start = 1

    def _get_total(self, runner) -> int:
        return runner.get_total_steps()

    def _training_loop(self, strategy_name, loop_state):
        runner = self.runner
        config = runner.config
        hook_manager = self.hook_manager
        s = loop_state
        c = s.components

        # RL-specific config
        rollout_length = getattr(config, 'rollout_length', 2048)
        update_epochs = getattr(config, 'update_epochs', 4)
        batch_size = getattr(config, 'batch_size', 64)
        env = c.data  # environment
        buffer = c.rollout_buffer

        if buffer is None:
            raise ValueError("RolloutTrainer requires components.rollout_buffer")
        if env is None:
            raise ValueError("RolloutTrainer requires components.data (environment)")

        # Initial eval
        if not s.is_resuming:
            init_eval = runner.evaluate(c.get_primary_model(), step_or_epoch=0)
            if init_eval:
                runner.display_eval(0, init_eval, strategy_name)
                if hook_manager:
                    hook_manager.emit_metrics(
                        init_eval.metrics, step=0, hook_point=HookPoint.SNAPSHOT,
                    )
        else:
            init_eval = None

        # Training
        display.training_progress_start(strategy_name, s.total)
        c.train_mode()

        from ..hooks.support import build_intervention_context_if_needed

        last_eval_result = None
        training_start = time.time()
        s.emergency.capture(c, s.start_counter, runner)

        env_step = s.start_counter
        global_update_step = 0
        obs = env.reset()

        try:
            while env_step <= s.total:
                # === Phase 1: Collect rollouts ===
                buffer.reset()
                c.eval_mode()

                for _ in range(rollout_length):
                    if env_step > s.total:
                        break

                    with torch.no_grad():
                        obs_tensor = (obs if isinstance(obs, torch.Tensor)
                                      else torch.tensor(obs, dtype=torch.float32))
                        obs_tensor = obs_tensor.to(c.get_device())

                        action_dist = c.actor(obs_tensor.unsqueeze(0))
                        action = action_dist.sample()
                        log_prob = action_dist.log_prob(action)
                        value = c.critic(obs_tensor.unsqueeze(0)).squeeze()

                    next_obs, reward, done, info = env.step(
                        action.squeeze(0).cpu().numpy()
                        if action.dim() > 0 else action.item()
                    )

                    buffer.add(
                        obs=obs_tensor.cpu(),
                        action=action.squeeze(0).cpu(),
                        reward=float(reward),
                        done=float(done),
                        log_prob=log_prob.squeeze(0).cpu(),
                        value=value.item(),
                    )

                    obs = next_obs
                    if done:
                        obs = env.reset()
                    env_step += 1

                if len(buffer) == 0:
                    break

                # Compute advantages
                with torch.no_grad():
                    last_obs = (obs if isinstance(obs, torch.Tensor)
                                else torch.tensor(obs, dtype=torch.float32))
                    last_obs = last_obs.to(c.get_device())
                    last_value = c.critic(last_obs.unsqueeze(0)).squeeze().item()
                buffer.compute_returns(last_value)

                # === Phase 2: Policy updates ===
                c.train_mode()

                for update_epoch in range(update_epochs):
                    for minibatch in buffer.get_batches(batch_size):
                        global_update_step += 1
                        result = c.strategy.train_step(global_update_step, batch=minibatch)

                        if result.trained:
                            c.step_schedulers()

                # POST_STEP hooks (fire once per rollout)
                if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_STEP):
                    ctx = RunContext(
                        hook_point=HookPoint.POST_STEP,
                        epoch=0, step=env_step,
                        profiler=s.profiler,
                    )
                    hook_manager.fire(
                        HookPoint.POST_STEP, ctx,
                        model_state=c.build_model_state(),
                        eval_metrics=EvalMetrics(
                            metrics={k: float(v) for k, v in result.metrics.items()
                                     if v is not None}
                        ),
                        gradient_state=c.build_gradient_state(),
                    )

                # Strategy metrics
                if hook_manager and result.metrics:
                    hook_manager.emit_metrics(
                        result.metrics, step=env_step,
                        hook_point=HookPoint.POST_STEP,
                    )

                # SNAPSHOT (periodic)
                if env_step % config.snapshot_every == 0:
                    if hook_manager:
                        ctx = RunContext(
                            hook_point=HookPoint.SNAPSHOT,
                            epoch=0, step=env_step,
                            config=config, profiler=s.profiler,
                        )
                        hook_manager.fire(
                            HookPoint.SNAPSHOT, ctx,
                            model_state=c.build_model_state(),
                            eval_metrics=EvalMetrics(
                                metrics={k: float(v) for k, v in result.metrics.items()
                                         if v is not None}
                            ),
                        )

                    s.emergency.capture(c, env_step, runner)

                # Evaluation (periodic)
                if env_step % config.eval_every == 0:
                    last_eval_result = runner.evaluate(c.get_primary_model(), env_step)
                    if last_eval_result:
                        runner.display_eval(env_step, last_eval_result, strategy_name)
                        if hook_manager:
                            hook_manager.emit_metrics(
                                last_eval_result.metrics, step=env_step,
                                hook_point=HookPoint.SNAPSHOT,
                            )

                # Checkpoint
                self._handle_periodic_checkpoint(env_step, s)

                # Progress
                display.training_progress_update(env_step, s.total, result, strategy_name)

                # Early stopping
                if env_step % config.eval_every == 0 and runner.should_stop(
                        env_step, last_eval_result):
                    break

        except KeyboardInterrupt:
            self._handle_interrupt(env_step, s)
            raise

        display.training_progress_end()
        c.strategy.teardown()
        duration = time.time() - training_start

        return TrainResult(
            final_step_or_epoch=env_step,
            init_eval=init_eval,
            duration=duration,
            early_stopped=(env_step < s.total),
        )
