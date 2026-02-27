"""Hook support helpers for the framework trainers.

Provides demand-driven construction helpers and intervention context
building. These are internal to the framework — experiments never call
them directly.

Trainers construct RunContext and state objects directly, then pass them
to HookManager.fire(). These helpers handle the conditional/expensive
operations: gradient accumulation, state capture, intervention context.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .hook_point import HookPoint
from ..contexts import BackpropInterventionContext
from ..utils import create_accumulator


def build_intervention_context_if_needed(hook_manager, hook_point, epoch, *,
                                          components=None,
                                          model=None, optimizer=None,
                                          scheduler=None, criterion=None,
                                          loader=None, config=None, device=None,
                                          pre_epoch_state=None,
                                          current_batch=None, profiler=None,
                                          loss_fn=None):
    """Build BackpropInterventionContext only if intervention hooks are active at this point.

    Accepts either a TrainingComponents bundle (via components=) or individual
    component args. When components is provided, its build_intervention_context_kwargs()
    is used and individual args override where specified.

    Returns None if no intervention hooks need it.
    """
    if hook_manager is None:
        return None

    needs_it = False
    if hook_point in (HookPoint.PRE_STEP, HookPoint.POST_STEP):
        needs_it = hook_manager.has_active_step_interventions(hook_point, epoch=epoch)
    elif hook_point in (HookPoint.PRE_EPOCH, HookPoint.POST_EPOCH, HookPoint.SNAPSHOT):
        needs_it = hook_manager.has_interventions_at(hook_point, epoch=epoch)

    if not needs_it:
        return None

    if components is not None:
        kwargs = components.build_intervention_context_kwargs(
            loader=loader, config=config, device=device,
            pre_epoch_state=pre_epoch_state,
            current_batch=current_batch,
            profiler=profiler,
        )
        return BackpropInterventionContext(**kwargs)

    return BackpropInterventionContext(
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
    import torch

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = None

    return {
        'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer': copy.deepcopy(optimizer.state_dict()),
        'rng_cpu': torch.random.get_rng_state(),
        'rng_cuda': (torch.cuda.get_rng_state(device)
                     if device is not None and device.type == 'cuda' else None),
    }
