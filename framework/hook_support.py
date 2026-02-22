"""Hook support helpers for the framework loops.

Reduce boilerplate in step_loop() and epoch_loop() by providing
builder functions for hook contexts and conditional setup. These
are internal to the framework — experiments never call them directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from training_hooks.base import HookPoint
from training_hooks.contexts import RunDataContext, ModelDataContext
from training_hooks.grad_accumulator import create_accumulator


def build_run_context(hook_point, *, epoch, model, config=None,
                      step=None, loader=None, batch_idx=None, batch_data=None,
                      loss=None, train_acc=None, val_acc=None, lr=None,
                      accumulated_grads=None, prev_step_grads=None,
                      target_grad=None, profiler=None):
    """Build a RunDataContext from loop state."""
    return RunDataContext(
        hook_point=hook_point,
        epoch=epoch,
        step=step,
        model=model,
        config=config,
        loader=loader,
        batch_idx=batch_idx,
        batch_data=batch_data,
        loss=loss,
        train_acc=train_acc,
        val_acc=val_acc,
        lr=lr,
        accumulated_grads=accumulated_grads,
        prev_step_grads=prev_step_grads,
        target_grad=target_grad,
        profiler=profiler,
    )


def build_model_context_if_needed(hook_manager, hook_point, epoch, *,
                                   model, optimizer, scheduler, criterion,
                                   loader, config, device,
                                   pre_epoch_state=None,
                                   current_batch=None, profiler=None,
                                   loss_fn=None):
    """Build ModelDataContext only if intervention hooks are active at this point.

    Returns None if no intervention hooks need it.
    """
    if hook_manager is None:
        return None

    needs_it = False
    if hook_point in (HookPoint.PRE_STEP, HookPoint.POST_STEP):
        needs_it = hook_manager.has_active_step_interventions(hook_point, epoch=epoch)
    elif hook_point in (HookPoint.POST_EPOCH, HookPoint.SNAPSHOT):
        # POST_EPOCH and SNAPSHOT intervention hooks need model context
        needs_it = hook_manager.has_hooks_at(hook_point, epoch=epoch)

    if not needs_it:
        return None

    return ModelDataContext(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        loader=loader,
        config=config,
        device=device,
        pre_epoch_state=pre_epoch_state,
        current_batch=current_batch,
        profiler=profiler,
        loss_fn=loss_fn,
    )


def setup_grad_accumulation_if_needed(hook_manager, model, config,
                                      is_snapshot, record_trajectory):
    """Create gradient accumulator only if hooks/trajectory need it.

    Returns (accumulator_dict, batch_count) or (None, 0).
    """
    needs_grads = (record_trajectory and is_snapshot)
    if hook_manager is not None:
        if is_snapshot and hook_manager.needs_grad_accumulation_at(
                HookPoint.SNAPSHOT):
            needs_grads = True
        if hook_manager.needs_grad_accumulation_at(HookPoint.POST_EPOCH):
            needs_grads = True

    if needs_grads:
        return create_accumulator(model)
    return None, 0


def capture_prev_step_grads_if_needed(hook_manager, model, epoch):
    """Capture previous step gradients if needed by POST_STEP hooks.

    Returns gradient dict or None.
    """
    if hook_manager is None:
        return None
    if not hook_manager.needs_prev_step_grads_this_step(epoch=epoch):
        return None

    # Capture current gradients before they get overwritten by the next step
    prev_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            prev_grads[name] = param.grad.detach().clone()
    return prev_grads if prev_grads else None


def capture_pre_epoch_state_if_needed(hook_manager, model, optimizer, epoch):
    """Capture pre-epoch model/optimizer state if any epoch-level hooks need it.

    Checks both POST_EPOCH and SNAPSHOT since intervention hooks may
    transition between them via epoch-gated loop_points.

    Returns state dict or None.
    """
    if hook_manager is None:
        return None
    if not (hook_manager.needs_pre_epoch_state_at(HookPoint.POST_EPOCH, epoch=epoch)
            or hook_manager.needs_pre_epoch_state_at(HookPoint.SNAPSHOT, epoch=epoch)):
        return None

    import copy
    return {
        'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
    }
