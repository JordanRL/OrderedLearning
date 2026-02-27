"""Adversarial (GAN) step-based trainer.

Mirrors StepTrainer but constructs adversarial-specific state objects
via components.build_model_state() / components.build_gradient_state().
"""

from __future__ import annotations

import time

from ..hooks import HookPoint
from ..capabilities import TrainingParadigm, GradientAvailability
from ..contexts import RunContext, EvalMetrics, BatchState
from .. import display
from .base import Trainer, LoopState, TrainResult


class AdversarialStepTrainer(Trainer):
    """Step-based adversarial training (GANs).

    Scheduler steps every training step. Uses AdversarialComponents
    for dual model/optimizer management.
    """

    paradigm = TrainingParadigm.ADVERSARIAL
    gradient_availability = GradientAvailability.GLOBAL_GRADIENTS
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

        if hasattr(runner, '_init_eval'):
            runner._init_eval = init_eval

        # Training
        display.training_progress_start(strategy_name, s.total)
        c.train_mode()

        from ..hooks.support import build_intervention_context_if_needed

        last_eval_result = None
        training_start = time.time()
        s.emergency.capture(c, s.start_counter, runner)
        step = s.start_counter

        try:
            for step in range(s.start_counter, s.total + 1):
                result = c.strategy.train_step(step)

                if result.trained:
                    c.step_schedulers()

                # POST_STEP hooks
                if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_STEP):
                    ctx = RunContext(
                        hook_point=HookPoint.POST_STEP,
                        epoch=0, step=step,
                        profiler=s.profiler,
                    )
                    model_ctx = build_intervention_context_if_needed(
                        hook_manager, HookPoint.POST_STEP, 0,
                        components=c, loader=c.data, config=config,
                        device=runner.device, current_batch=result.batch_data,
                        profiler=s.profiler,
                    )
                    hook_manager.fire(
                        HookPoint.POST_STEP, ctx,
                        model_ctx=model_ctx,
                        model_state=c.build_model_state(),
                        eval_metrics=EvalMetrics(
                            metrics={k: float(v) for k, v in result.metrics.items()
                                     if v is not None}
                        ),
                        gradient_state=c.build_gradient_state(),
                        batch_state=BatchState(batch_idx=step, batch_data=result.batch_data),
                    )

                # Strategy metrics
                if hook_manager and result.metrics:
                    hook_manager.emit_metrics(
                        result.metrics, step=step, hook_point=HookPoint.POST_STEP,
                    )

                # SNAPSHOT (periodic)
                if step % config.snapshot_every == 0:
                    if hook_manager:
                        ctx = RunContext(
                            hook_point=HookPoint.SNAPSHOT,
                            epoch=0, step=step,
                            config=config, profiler=s.profiler,
                        )
                        model_ctx = build_intervention_context_if_needed(
                            hook_manager, HookPoint.SNAPSHOT, 0,
                            components=c, loader=c.data, config=config,
                            device=runner.device, profiler=s.profiler,
                        )
                        hook_manager.fire(
                            HookPoint.SNAPSHOT, ctx,
                            model_ctx=model_ctx,
                            model_state=c.build_model_state(),
                            eval_metrics=EvalMetrics(
                                metrics={k: float(v) for k, v in result.metrics.items()
                                         if v is not None}
                            ),
                        )

                    if s.trajectory is not None:
                        from ..utils import snapshot_params
                        s.trajectory.append({
                            'step': step,
                            'params': snapshot_params(c.get_primary_model()),
                            'g_loss': result.metrics.get('g_loss'),
                            'd_loss': result.metrics.get('d_loss'),
                            'strategy': strategy_name,
                        })

                    s.emergency.capture(c, step, runner)

                # Evaluation (periodic)
                if step % config.eval_every == 0:
                    last_eval_result = runner.evaluate(c.get_primary_model(), step)
                    if last_eval_result:
                        runner.display_eval(step, last_eval_result, strategy_name)
                        if hook_manager:
                            hook_manager.emit_metrics(
                                last_eval_result.metrics, step=step,
                                hook_point=HookPoint.SNAPSHOT,
                            )

                # Checkpoint (periodic)
                self._handle_periodic_checkpoint(step, s)

                # Progress
                display.training_progress_update(step, s.total, result, strategy_name)

                # Control flow
                if result.should_stop:
                    break
                if step % config.eval_every == 0 and runner.should_stop(
                        step, last_eval_result):
                    break

        except KeyboardInterrupt:
            self._handle_interrupt(step, s)
            raise

        display.training_progress_end()
        c.strategy.teardown()
        duration = time.time() - training_start

        return TrainResult(
            final_step_or_epoch=step,
            init_eval=init_eval,
            duration=duration,
            early_stopped=(step < s.total),
        )
