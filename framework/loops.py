"""Framework training loops.

step_loop() and epoch_loop() compose calls to ExperimentRunner and
StrategyRunner methods. Experiments never write these loops — they
implement the building blocks that the loops call.

The fundamental difference between the two loops:
- step_loop: scheduler.step() after every training step
- epoch_loop: scheduler.step() after every epoch (full pass through data)
"""

from __future__ import annotations

import time

from training_hooks.base import HookPoint
from training_hooks.grad_accumulator import accumulate, finalize

from .utils import snapshot_params, get_environment_info
from . import display
from .hook_support import (
    build_run_context,
    build_model_context_if_needed,
    setup_grad_accumulation_if_needed,
    capture_prev_step_grads_if_needed,
    capture_pre_epoch_state_if_needed,
)


def _get_rng_states() -> dict:
    """Capture current RNG states for all relevant backends."""
    import random
    import numpy as np
    import torch

    states = {
        'torch': torch.random.get_rng_state(),
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }
    if torch.cuda.is_available():
        states['torch_cuda'] = torch.cuda.get_rng_state_all()
    return states


def save_checkpoint(experiment_dir, model, optimizer, scheduler, step_or_epoch,
                    training_state=None) -> None:
    """Save training checkpoint (includes environment metadata and RNG states)."""
    import os
    import torch
    ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'checkpoint_{step_or_epoch}.pt')
    torch.save({
        'step': step_or_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'rng_states': _get_rng_states(),
        'training_state': training_state,
        'environment': get_environment_info(),
    }, path)


def _compare_state(live, saved) -> bool:
    """Recursively compare two state structures for bit-identity.

    Handles tensors (torch.equal), numpy arrays, dicts, lists/tuples,
    and scalars. Returns True if every value matches exactly.
    """
    import numpy as np
    import torch

    if type(live) is not type(saved):
        return False
    if isinstance(live, torch.Tensor):
        if live.shape != saved.shape or live.dtype != saved.dtype:
            return False
        return torch.equal(live.cpu(), saved.cpu())
    if isinstance(live, np.ndarray):
        return np.array_equal(live, saved)
    if isinstance(live, dict):
        if live.keys() != saved.keys():
            return False
        return all(_compare_state(live[k], saved[k]) for k in live)
    if isinstance(live, (list, tuple)):
        if len(live) != len(saved):
            return False
        return all(_compare_state(a, b) for a, b in zip(live, saved))
    return live == saved


def validate_checkpoint(experiment_dir, model, optimizer, scheduler, step_or_epoch,
                        training_state=None) -> None:
    """Compare current training state against a saved checkpoint.

    Loads the checkpoint at the given step/epoch and compares model,
    optimizer, scheduler state dicts, RNG states, and experiment-specific
    training state for bit-identity. Prints a verification result through
    OLConsole.
    """
    import os
    import torch
    from console import OLConsole

    console = OLConsole()
    path = os.path.join(experiment_dir, 'checkpoints', f'checkpoint_{step_or_epoch}.pt')

    if not os.path.exists(path):
        console.print_warning(f"No checkpoint found at step/epoch {step_or_epoch}")
        return

    # weights_only=False is required for optimizer/scheduler state dicts
    # and RNG states. See security note in framework/resume.py.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    model_ok = _compare_state(model.state_dict(), checkpoint['model_state_dict'])
    optim_ok = _compare_state(optimizer.state_dict(), checkpoint['optimizer_state_dict'])

    sched_ok = True
    if scheduler and checkpoint.get('scheduler_state_dict'):
        sched_ok = _compare_state(scheduler.state_dict(), checkpoint['scheduler_state_dict'])

    rng_ok = True
    has_rng = 'rng_states' in checkpoint
    if has_rng:
        rng_ok = _compare_state(_get_rng_states(), checkpoint['rng_states'])

    state_ok = True
    has_state = checkpoint.get('training_state') is not None
    if has_state and training_state is not None:
        state_ok = _compare_state(training_state, checkpoint['training_state'])

    all_ok = model_ok and optim_ok and sched_ok and rng_ok and state_ok

    def _mark(ok):
        return '[metric.improved]OK[/metric.improved]' if ok else '[metric.degraded]MISMATCH[/metric.degraded]'

    rng_display = f"  rng {_mark(rng_ok)}" if has_rng else ""
    state_display = f"  state {_mark(state_ok)}" if has_state else ""
    label = step_or_epoch
    if all_ok:
        console.print_complete(
            f"Checkpoint {label}: [metric.improved]bit-identical[/metric.improved]"
            f"  (model {_mark(model_ok)}  optimizer {_mark(optim_ok)}  scheduler {_mark(sched_ok)}{rng_display}{state_display})"
        )
    else:
        console.print_error(
            f"Checkpoint {label}: non-identical"
            f"  (model {_mark(model_ok)}  optimizer {_mark(optim_ok)}  scheduler {_mark(sched_ok)}{rng_display}{state_display})"
        )


def step_loop(runner, hook_manager, resume=None) -> dict:
    """Step-based training loop. Scheduler steps every training step.

    Used by: presorted, phased_curriculum, guided_llm.
    """
    config = runner.config
    runner.display_banner()

    all_results = {}

    for strategy_name in runner.get_strategies():

        if resume and strategy_name in resume.completed_strategies:
            continue

        # ── SETUP ──
        runner.setup_condition(strategy_name)
        runner.display_condition_start(strategy_name)

        model = runner.create_model()
        data = runner.create_data(strategy_name)
        strategy = runner.create_strategy(strategy_name)
        optimizer = runner.create_optimizer(model)
        total_steps = runner.get_total_steps()
        scheduler = runner.create_scheduler(optimizer, total_steps)
        criterion = runner.create_criterion()

        strategy_kwargs = runner.get_strategy_kwargs(
            strategy_name, model, optimizer, data
        )
        strategy.setup(
            model=model, optimizer=optimizer, config=config,
            device=runner.device, data=data, criterion=criterion,
            **strategy_kwargs,
        )

        if hook_manager:
            hook_manager.reset_all()
            hook_manager.set_run_context(strategy=strategy_name)

        runner.wire_hooks(strategy_name, strategy, hook_manager)

        experiment_dir = runner.prepare_output_dir(strategy_name)
        runner.save_config(experiment_dir, extra={'strategy': strategy_name})
        trajectory = [] if config.record_trajectory else None

        profiler = hook_manager.profiler if hook_manager else None
        loss_fn = runner.get_loss_fn(criterion)

        # ── RESUME: checkpoint loading ──
        is_resuming = (resume is not None
                       and resume.checkpoint_path is not None
                       and strategy_name not in resume.completed_strategies)

        if is_resuming:
            from .resume import load_checkpoint
            load_checkpoint(resume.checkpoint_path, model, optimizer,
                            scheduler, runner)
            display.display_resume_info(
                resume.start_step_or_epoch,
                resume.checkpoint_path,
                resume.completed_strategies,
            )

        # ── INITIAL EVAL ──
        if not is_resuming:
            init_eval = runner.evaluate(model, step_or_epoch=0)
            if init_eval:
                runner.display_eval(0, init_eval, strategy_name)
                if hook_manager:
                    hook_manager.emit_metrics(
                        init_eval.metrics, step=0, hook_point=HookPoint.SNAPSHOT,
                    )
        else:
            init_eval = None

        # Store for display_eval change tracking
        if hasattr(runner, '_init_eval'):
            runner._init_eval = init_eval

        # ── TRAINING LOOP ──
        display.training_progress_start(strategy_name, total_steps)
        model.train()

        start_step = resume.start_step_or_epoch + 1 if is_resuming else 1
        if is_resuming:
            resume = None  # consumed — remaining strategies run fresh

        last_eval_result = None
        training_start = time.time()
        try:
            for step in range(start_step, total_steps + 1):

                # --- Train ---
                result = strategy.train_step(step)

                # --- Scheduler ---
                if result.trained:
                    scheduler.step()

                # --- Hooks: POST_STEP ---
                if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_STEP):
                    run_ctx = build_run_context(
                        HookPoint.POST_STEP,
                        epoch=0, step=step, batch_idx=step,
                        batch_data=result.batch_data, model=model,
                        loss=result.loss, lr=scheduler.get_last_lr()[0],
                        profiler=profiler,
                        target_grad=getattr(result, 'target_grad', None),
                    )
                    model_ctx = build_model_context_if_needed(
                        hook_manager, HookPoint.POST_STEP, epoch=0,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        criterion=criterion, loader=data, config=config,
                        device=runner.device, current_batch=result.batch_data,
                        loss_fn=loss_fn, profiler=profiler,
                    )
                    hook_manager.fire(HookPoint.POST_STEP, run_ctx, model_ctx)

                # --- Emit strategy-specific metrics ---
                if hook_manager and result.metrics:
                    hook_manager.emit_metrics(
                        result.metrics, step=step, hook_point=HookPoint.POST_STEP,
                    )

                # --- Strategy post-step (e.g. phase transition) ---
                post_info = strategy.post_step(step, result)
                if post_info:
                    runner.display_post_step(step, post_info)
                    if hook_manager and isinstance(post_info, dict):
                        numeric = {k: v for k, v in post_info.items()
                                   if isinstance(v, (int, float))}
                        if numeric:
                            hook_manager.emit_metrics(
                                numeric, step=step, hook_point=HookPoint.POST_STEP,
                            )

                # --- Hooks: SNAPSHOT (periodic) ---
                if step % config.snapshot_every == 0:
                    if hook_manager:
                        run_ctx = build_run_context(
                            HookPoint.SNAPSHOT,
                            epoch=0, step=step, model=model,
                            accumulated_grads=None,
                            loss=result.loss, train_acc=0.0, val_acc=0.0,
                            lr=scheduler.get_last_lr()[0], config=config,
                            profiler=profiler,
                        )
                        hook_manager.fire(HookPoint.SNAPSHOT, run_ctx)

                    if trajectory is not None:
                        trajectory.append({
                            'step': step,
                            'params': snapshot_params(model),
                            'loss': result.loss,
                            'strategy': strategy_name,
                        })

                # --- Evaluation (periodic) ---
                if step % config.eval_every == 0:
                    last_eval_result = runner.evaluate(model, step)
                    if last_eval_result:
                        runner.display_eval(step, last_eval_result, strategy_name)
                        if hook_manager:
                            hook_manager.emit_metrics(
                                last_eval_result.metrics, step=step,
                                hook_point=HookPoint.SNAPSHOT,
                            )

                # --- Checkpoint (periodic) ---
                if step % config.checkpoint_every == 0:
                    training_state = runner.save_training_state()
                    if config.save_checkpoints:
                        save_checkpoint(experiment_dir, model, optimizer, scheduler, step,
                                        training_state=training_state)
                    elif config.validate_checkpoints:
                        validate_checkpoint(experiment_dir, model, optimizer, scheduler, step,
                                            training_state=training_state)

                # --- Progress ---
                display.training_progress_update(step, total_steps, result, strategy_name)

                # --- Control flow ---
                if result.should_stop:
                    break
                if step % config.eval_every == 0 and runner.should_stop(
                        step, last_eval_result):
                    break

        except KeyboardInterrupt:
            from console import OLConsole
            _console = OLConsole()
            _console.print_warning(
                f"Training interrupted at step {step}. Saving emergency checkpoint..."
            )
            training_state = runner.save_training_state()
            save_checkpoint(experiment_dir, model, optimizer, scheduler, step,
                            training_state=training_state)
            _console.print_complete(
                f"Emergency checkpoint saved at step {step} in {experiment_dir}/checkpoints/"
            )
            if hook_manager:
                hook_manager.flush_sinks()
            raise

        display.training_progress_end()
        strategy.teardown()
        duration = time.time() - training_start

        # ── FINALIZE ──
        final_eval = runner.evaluate(model, step)
        if final_eval:
            runner.display_final(strategy_name, init_eval, final_eval)

        summary = runner.build_summary(
            strategy_name, init_eval, final_eval, step,
            model=model, planned_total=total_steps, duration=duration,
            early_stopped=(step < total_steps),
        )
        runner.save_summary(experiment_dir, summary)
        runner.save_trajectory(experiment_dir, trajectory)
        runner.save_final_model(experiment_dir, model, strategy_name)

        # Completion checkpoint — always saved for resume support
        training_state = runner.save_training_state()
        save_checkpoint(experiment_dir, model, optimizer, scheduler, step,
                        training_state=training_state)

        runner.teardown_condition(strategy_name)
        if hook_manager:
            hook_manager.flush_sinks()

        all_results[strategy_name] = summary

    runner.display_comparison(all_results)
    return all_results


def epoch_loop(runner, hook_manager, resume=None) -> dict:
    """Epoch-based training loop. Scheduler steps every epoch.

    Used by: mod_arithmetic.
    """
    config = runner.config
    runner.display_banner()

    all_results = {}

    for strategy_name in runner.get_strategies():

        if resume and strategy_name in resume.completed_strategies:
            continue

        # ── SETUP ──
        runner.setup_condition(strategy_name)
        runner.display_condition_start(strategy_name)

        model = runner.create_model()
        data = runner.create_data(strategy_name)
        strategy = runner.create_strategy(strategy_name)
        optimizer = runner.create_optimizer(model)
        total_epochs = runner.get_total_epochs()
        scheduler = runner.create_scheduler(optimizer, total_epochs)
        criterion = runner.create_criterion()

        strategy_kwargs = runner.get_strategy_kwargs(
            strategy_name, model, optimizer, data
        )
        strategy.setup(
            model=model, optimizer=optimizer, config=config,
            device=runner.device, data=data, criterion=criterion,
            **strategy_kwargs,
        )

        if hook_manager:
            hook_manager.reset_all()
            hook_manager.set_run_context(strategy=strategy_name)

        runner.wire_hooks(strategy_name, strategy, hook_manager)

        experiment_dir = runner.prepare_output_dir(strategy_name)
        runner.save_config(experiment_dir, extra={'strategy': strategy_name})
        trajectory = [] if config.record_trajectory else None

        profiler = hook_manager.profiler if hook_manager else None
        loss_fn = runner.get_loss_fn(criterion)

        # ── RESUME: checkpoint loading ──
        is_resuming = (resume is not None
                       and resume.checkpoint_path is not None
                       and strategy_name not in resume.completed_strategies)

        if is_resuming:
            from .resume import load_checkpoint
            load_checkpoint(resume.checkpoint_path, model, optimizer,
                            scheduler, runner)
            display.display_resume_info(
                resume.start_step_or_epoch,
                resume.checkpoint_path,
                resume.completed_strategies,
                counter_label="epoch",
            )

        # ── INITIAL EVAL ──
        if not is_resuming:
            init_eval = runner.evaluate(model, step_or_epoch=0)
        else:
            init_eval = None

        # ── TRAINING LOOP ──
        display.epoch_progress_start(strategy_name, total_epochs)

        start_epoch = resume.start_step_or_epoch + 1 if is_resuming else 0
        global_step = start_epoch * len(runner.get_epoch_loader(data, 0)) if is_resuming else 0
        if is_resuming:
            resume = None  # consumed — remaining strategies run fresh

        training_start = time.time()
        try:
            for epoch in range(start_epoch, total_epochs):
                loader = runner.get_epoch_loader(data, epoch)
                # Store for GrokkingRunner.evaluate()
                if hasattr(runner, '_current_train_loader'):
                    runner._current_train_loader = loader
                model.train()

                # Hook bookkeeping
                is_snapshot = (epoch % config.snapshot_every == 0)
                is_eval = is_snapshot or (epoch % config.eval_every == 0)

                grad_accum, grad_count = setup_grad_accumulation_if_needed(
                    hook_manager, model, config, is_snapshot,
                    config.record_trajectory,
                )

                pre_epoch_state = capture_pre_epoch_state_if_needed(
                    hook_manager, model, optimizer, epoch,
                )

                # PRE_EPOCH hooks
                if hook_manager and hook_manager.has_hooks_at(HookPoint.PRE_EPOCH, epoch=epoch):
                    run_ctx = build_run_context(
                        HookPoint.PRE_EPOCH,
                        epoch=epoch, model=model, loader=loader, config=config,
                        profiler=profiler,
                    )
                    hook_manager.fire(HookPoint.PRE_EPOCH, run_ctx)

                # ── BATCH LOOP ──
                display.batch_progress_start(len(loader))
                epoch_loss = 0
                batch_count = 0

                for batch_idx, batch in enumerate(loader):
                    global_step += 1
                    if hook_manager:
                        hook_manager.advance_step()

                    # PRE_STEP hooks
                    prev_grads = capture_prev_step_grads_if_needed(
                        hook_manager, model, epoch,
                    )
                    if hook_manager and hook_manager.has_hooks_at(HookPoint.PRE_STEP, epoch=epoch):
                        run_ctx = build_run_context(
                            HookPoint.PRE_STEP,
                            epoch=epoch, step=global_step, batch_idx=batch_idx,
                            batch_data=batch, model=model, profiler=profiler,
                        )
                        hook_manager.fire(HookPoint.PRE_STEP, run_ctx)

                    # TRAIN
                    result = strategy.train_step(global_step, batch=batch)
                    epoch_loss += result.loss
                    batch_count += 1

                    # Gradient accumulation
                    if grad_accum is not None:
                        grad_count = accumulate(grad_accum, model, grad_count)

                    # POST_STEP hooks
                    if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_STEP, epoch=epoch):
                        run_ctx = build_run_context(
                            HookPoint.POST_STEP,
                            epoch=epoch, step=global_step, batch_idx=batch_idx,
                            batch_data=result.batch_data, model=model,
                            loss=result.loss, prev_step_grads=prev_grads,
                            profiler=profiler,
                        )
                        model_ctx = build_model_context_if_needed(
                            hook_manager, HookPoint.POST_STEP, epoch,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            criterion=criterion, loader=loader, config=config,
                            device=runner.device, current_batch=batch,
                            profiler=profiler, loss_fn=loss_fn,
                        )
                        hook_manager.fire(HookPoint.POST_STEP, run_ctx, model_ctx)

                    display.batch_progress_update()

                display.batch_progress_end()

                # Finalize gradients
                if hook_manager:
                    hook_manager.flush_step_metrics(epoch)
                if grad_accum is not None and grad_count > 0:
                    grad_accum = finalize(grad_accum, grad_count, to_cpu=False)

                avg_loss = float(epoch_loss) / max(batch_count, 1)

                # SCHEDULER (per epoch)
                scheduler.step()

                # EVALUATION (periodic)
                eval_result = None
                if is_eval:
                    eval_result = runner.evaluate(model, epoch)
                    if eval_result and is_snapshot:
                        runner.display_eval(epoch, eval_result, strategy_name)

                # POST_EPOCH hooks
                if hook_manager and hook_manager.has_hooks_at(HookPoint.POST_EPOCH, epoch=epoch):
                    run_ctx = build_run_context(
                        HookPoint.POST_EPOCH,
                        epoch=epoch, model=model, loader=loader, config=config,
                        loss=avg_loss,
                        train_acc=eval_result.metrics.get('train_acc') if eval_result else None,
                        val_acc=eval_result.metrics.get('val_acc') if eval_result else None,
                        lr=optimizer.param_groups[0]['lr'],
                        accumulated_grads=grad_accum,
                        profiler=profiler,
                    )
                    model_ctx = build_model_context_if_needed(
                        hook_manager, HookPoint.POST_EPOCH, epoch,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        criterion=criterion, loader=loader, config=config,
                        device=runner.device,
                        pre_epoch_state=pre_epoch_state,
                        profiler=profiler, loss_fn=loss_fn,
                    )
                    hook_manager.fire(HookPoint.POST_EPOCH, run_ctx, model_ctx)

                # SNAPSHOT hooks + trajectory (periodic)
                if is_snapshot:
                    if hook_manager:
                        run_ctx = build_run_context(
                            HookPoint.SNAPSHOT,
                            epoch=epoch, step=global_step, model=model,
                            accumulated_grads=grad_accum,
                            loss=avg_loss,
                            train_acc=eval_result.metrics.get('train_acc', 0.0) if eval_result else 0.0,
                            val_acc=eval_result.metrics.get('val_acc', 0.0) if eval_result else 0.0,
                            lr=optimizer.param_groups[0]['lr'], config=config,
                            profiler=profiler,
                        )
                        model_ctx = build_model_context_if_needed(
                            hook_manager, HookPoint.SNAPSHOT, epoch,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            criterion=criterion, loader=loader, config=config,
                            device=runner.device,
                            pre_epoch_state=pre_epoch_state,
                            profiler=profiler, loss_fn=loss_fn,
                        )
                        hook_manager.fire(HookPoint.SNAPSHOT, run_ctx, model_ctx)

                    if trajectory is not None:
                        trajectory.append({
                            'epoch': epoch,
                            'params': snapshot_params(model),
                            'grads': {n: g.cpu().clone() for n, g in grad_accum.items()} if grad_accum else {},
                            'eval': eval_result.metrics if eval_result else None,
                        })

                # Cleanup
                del grad_accum

                # Checkpoint
                if epoch % config.checkpoint_every == 0:
                    training_state = runner.save_training_state()
                    if config.save_checkpoints:
                        save_checkpoint(experiment_dir, model, optimizer, scheduler, epoch,
                                        training_state=training_state)
                    elif config.validate_checkpoints:
                        validate_checkpoint(experiment_dir, model, optimizer, scheduler, epoch,
                                            training_state=training_state)

                display.epoch_progress_update(epoch, total_epochs, avg_loss, eval_result,
                                                 strategy_name=strategy_name)

                # Early stopping
                if runner.should_stop(epoch, eval_result):
                    break

        except KeyboardInterrupt:
            from console import OLConsole
            _console = OLConsole()
            _console.print_warning(
                f"Training interrupted at epoch {epoch}. Saving emergency checkpoint..."
            )
            training_state = runner.save_training_state()
            save_checkpoint(experiment_dir, model, optimizer, scheduler, epoch,
                            training_state=training_state)
            _console.print_complete(
                f"Emergency checkpoint saved at epoch {epoch} in {experiment_dir}/checkpoints/"
            )
            if hook_manager:
                hook_manager.flush_sinks()
            raise

        display.epoch_progress_end()
        strategy.teardown()
        duration = time.time() - training_start

        # ── FINALIZE ──
        final_eval = runner.evaluate(model, epoch)
        if final_eval:
            runner.display_final(strategy_name, init_eval, final_eval)

        summary = runner.build_summary(
            strategy_name, init_eval, final_eval, epoch,
            model=model, planned_total=total_epochs, duration=duration,
            early_stopped=(epoch < total_epochs - 1),
            global_step=global_step,
        )
        runner.save_summary(experiment_dir, summary)
        runner.save_trajectory(experiment_dir, trajectory)
        runner.save_final_model(experiment_dir, model, strategy_name)

        # Completion checkpoint — always saved for resume support
        training_state = runner.save_training_state()
        save_checkpoint(experiment_dir, model, optimizer, scheduler, epoch,
                        training_state=training_state)

        runner.teardown_condition(strategy_name)
        if hook_manager:
            hook_manager.flush_sinks()

        all_results[strategy_name] = summary

    runner.display_comparison(all_results)
    return all_results
