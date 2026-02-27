"""Epoch-based trainer. Scheduler steps every epoch."""

from __future__ import annotations

import time

from ..hooks import HookPoint

from ..capabilities import TrainingParadigm, GradientAvailability
from ..contexts import RunContext, EvalMetrics, BatchState
from .. import display
from .base import Trainer, LoopState, TrainResult


class EpochTrainer(Trainer):
    """Epoch-based training. Scheduler steps every epoch.

    Used by: mod_arithmetic.
    """

    paradigm = TrainingParadigm.BACKPROP
    gradient_availability = GradientAvailability.GLOBAL_GRADIENTS
    loop_type = 'epoch'
    _counter_label = "epoch"
    _default_start = 0

    def _get_total(self, runner) -> int:
        return runner.get_total_epochs()

    def _training_loop(self, strategy_name, loop_state):
        runner = self.runner
        config = runner.config
        hook_manager = self.hook_manager
        s = loop_state
        c = s.components

        from ..hooks.support import (
            build_intervention_context_if_needed,
            setup_grad_accumulation_if_needed,
            capture_prev_step_grads_if_needed,
            capture_pre_epoch_state_if_needed,
        )
        from ..utils import snapshot_params
        from ..utils import accumulate, finalize

        # Initial eval
        if not s.is_resuming:
            init_eval = runner.evaluate(c.get_primary_model(), step_or_epoch=0)
        else:
            init_eval = None

        # Training
        display.epoch_progress_start(strategy_name, s.total)

        if s.is_resuming:
            try:
                global_step = s.start_counter * len(runner.get_epoch_loader(c.data, 0))
            except TypeError:
                global_step = 0  # loader doesn't support len()
        else:
            global_step = 0
        epoch = s.start_counter

        training_start = time.time()
        try:
            for epoch in range(s.start_counter, s.total):
                s.emergency.capture(c, epoch, runner)
                loader = runner.get_epoch_loader(c.data, epoch)
                if hasattr(runner, '_current_train_loader'):
                    runner._current_train_loader = loader
                c.train_mode()

                # Hook bookkeeping
                is_snapshot = (epoch % config.snapshot_every == 0)
                is_eval = is_snapshot or (epoch % config.eval_every == 0)

                grad_accum, grad_count = setup_grad_accumulation_if_needed(
                    hook_manager, c.get_primary_model(), config, is_snapshot,
                    config.record_trajectory,
                )

                pre_epoch_state = capture_pre_epoch_state_if_needed(
                    hook_manager, c.get_primary_model(), c.optimizer, epoch,
                )

                # PRE_EPOCH hooks
                if hook_manager and hook_manager.has_hooks_at(HookPoint.PRE_EPOCH, epoch=epoch):
                    ctx = RunContext(
                        hook_point=HookPoint.PRE_EPOCH,
                        epoch=epoch, config=config, profiler=s.profiler,
                    )
                    model_ctx = build_intervention_context_if_needed(
                        hook_manager, HookPoint.PRE_EPOCH, epoch,
                        components=c, loader=loader, config=config,
                        device=runner.device,
                        pre_epoch_state=pre_epoch_state,
                        profiler=s.profiler,
                    )
                    hook_manager.fire(
                        HookPoint.PRE_EPOCH, ctx,
                        model_ctx=model_ctx,
                        model_state=c.build_model_state(),
                        batch_state=BatchState(loader=loader),
                    )

                # Batch loop
                try:
                    loader_len = len(loader)
                except TypeError:
                    loader_len = None
                display.batch_progress_start(loader_len)
                epoch_loss = 0
                batch_count = 0

                for batch_idx, batch in enumerate(loader):
                    global_step += 1
                    if hook_manager:
                        hook_manager.advance_step()

                    # PRE_STEP hooks
                    prev_grads = capture_prev_step_grads_if_needed(
                        hook_manager, c.get_primary_model(), epoch,
                    )
                    if hook_manager and hook_manager.has_hooks_at(HookPoint.PRE_STEP, epoch=epoch):
                        ctx = RunContext(
                            hook_point=HookPoint.PRE_STEP,
                            epoch=epoch, step=global_step,
                            profiler=s.profiler,
                        )
                        model_ctx = build_intervention_context_if_needed(
                            hook_manager, HookPoint.PRE_STEP, epoch,
                            components=c, loader=loader, config=config,
                            device=runner.device, current_batch=batch,
                            profiler=s.profiler,
                        )
                        hook_manager.fire(
                            HookPoint.PRE_STEP, ctx,
                            model_ctx=model_ctx,
                            model_state=c.build_model_state(),
                            batch_state=BatchState(batch_idx=batch_idx, batch_data=batch),
                        )

                    # Train
                    result = c.strategy.train_step(global_step, batch=batch)
                    epoch_loss += result.loss
                    batch_count += 1

                    # Gradient accumulation
                    if grad_accum is not None:
                        grad_count = accumulate(grad_accum, c.get_primary_model(), grad_count)

                    # POST_STEP hooks
                    if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_STEP, epoch=epoch):
                        ctx = RunContext(
                            hook_point=HookPoint.POST_STEP,
                            epoch=epoch, step=global_step,
                            profiler=s.profiler,
                        )
                        model_ctx = build_intervention_context_if_needed(
                            hook_manager, HookPoint.POST_STEP, epoch,
                            components=c, loader=loader, config=config,
                            device=runner.device, current_batch=batch,
                            profiler=s.profiler,
                        )
                        hook_manager.fire(
                            HookPoint.POST_STEP, ctx,
                            model_ctx=model_ctx,
                            model_state=c.build_model_state(),
                            eval_metrics=EvalMetrics(metrics={'loss': float(result.loss)}),
                            gradient_state=c.build_gradient_state(prev_step_grads=prev_grads),
                            batch_state=BatchState(batch_idx=batch_idx, batch_data=result.batch_data),
                        )

                    display.batch_progress_update()

                display.batch_progress_end()

                # Finalize gradients
                if hook_manager:
                    hook_manager.flush_step_metrics(epoch)
                if grad_accum is not None and grad_count > 0:
                    grad_accum = finalize(grad_accum, grad_count, to_cpu=False)

                avg_loss = float(epoch_loss) / max(batch_count, 1)

                # Scheduler (per epoch)
                c.step_schedulers()

                # Evaluation (periodic)
                eval_result = None
                if is_eval:
                    eval_result = runner.evaluate(c.get_primary_model(), epoch)
                    if eval_result and is_snapshot:
                        runner.display_eval(epoch, eval_result, strategy_name)

                # Build eval metrics for hooks
                eval_dict = {'loss': avg_loss}
                if eval_result:
                    eval_dict.update(eval_result.metrics)

                # POST_EPOCH hooks
                if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_EPOCH, epoch=epoch):
                    ctx = RunContext(
                        hook_point=HookPoint.POST_EPOCH,
                        epoch=epoch, config=config, profiler=s.profiler,
                    )
                    model_ctx = build_intervention_context_if_needed(
                        hook_manager, HookPoint.POST_EPOCH, epoch,
                        components=c, loader=loader, config=config,
                        device=runner.device,
                        pre_epoch_state=pre_epoch_state,
                        profiler=s.profiler,
                    )
                    hook_manager.fire(
                        HookPoint.POST_EPOCH, ctx,
                        model_ctx=model_ctx,
                        model_state=c.build_model_state(),
                        eval_metrics=EvalMetrics(metrics=eval_dict),
                        gradient_state=c.build_gradient_state(accumulated_grads=grad_accum),
                        batch_state=BatchState(loader=loader),
                    )

                # SNAPSHOT hooks + trajectory (periodic)
                if is_snapshot:
                    if hook_manager:
                        ctx = RunContext(
                            hook_point=HookPoint.SNAPSHOT,
                            epoch=epoch, step=global_step,
                            config=config, profiler=s.profiler,
                        )
                        model_ctx = build_intervention_context_if_needed(
                            hook_manager, HookPoint.SNAPSHOT, epoch,
                            components=c, loader=loader, config=config,
                            device=runner.device,
                            pre_epoch_state=pre_epoch_state,
                            profiler=s.profiler,
                        )
                        hook_manager.fire(
                            HookPoint.SNAPSHOT, ctx,
                            model_ctx=model_ctx,
                            model_state=c.build_model_state(),
                            eval_metrics=EvalMetrics(metrics=eval_dict),
                            gradient_state=c.build_gradient_state(accumulated_grads=grad_accum),
                        )

                    if s.trajectory is not None:
                        s.trajectory.append({
                            'epoch': epoch,
                            'params': snapshot_params(c.get_primary_model()),
                            'grads': {n: g.cpu().clone() for n, g in grad_accum.items()} if grad_accum else {},
                            'eval': eval_result.metrics if eval_result else None,
                        })

                # Cleanup
                del grad_accum

                # Checkpoint
                self._handle_periodic_checkpoint(epoch, s)

                display.epoch_progress_update(epoch, s.total, avg_loss, eval_result,
                                              strategy_name=strategy_name,
                                              progress_metric=runner.progress_metric)

                # Early stopping
                if runner.should_stop(epoch, eval_result):
                    break

        except KeyboardInterrupt:
            self._handle_interrupt(epoch, s)
            raise

        display.epoch_progress_end()
        c.strategy.teardown()
        duration = time.time() - training_start

        return TrainResult(
            final_step_or_epoch=epoch,
            init_eval=init_eval,
            duration=duration,
            early_stopped=(epoch < s.total - 1),
            global_step=global_step,
        )
