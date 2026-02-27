"""
HookManager: orchestrates hook execution during training.

Instantiates hooks, fires them at lifecycle points, collects metrics,
dispatches to sinks, and optionally manages state offloading.
"""

import json
import os
from typing import Any, TYPE_CHECKING

import torch

from console import OLConsole
from console.utils import apply_style

from .hook_point import HookPoint, STEP_HOOK_POINTS
from .training_hook import TrainingHook
from .intervention_hook import InterventionHook
from .registry import HookRegistry
from ..capabilities import HookNeeds
from ..utils import _json_default

if TYPE_CHECKING:
    from ..contexts import RunContext, BackpropInterventionContext, CheckpointProfiler
    from ..capabilities import TrainingCapabilities


class HookManager:
    """Orchestrates hook execution during training.

    Instantiates hooks, fires them at lifecycle points, collects metrics,
    dispatches to sinks, and optionally manages state offloading.
    """

    def __init__(
        self,
        hook_names: list[str] | None = None,
        hooks: list[TrainingHook] | None = None,
        sinks: list | None = None,
        offload_state: bool = False,
        hook_config: dict[str, dict[str, Any]] | None = None,
        step_metrics_log: str | None = "raw_logs/step_metrics.jsonl",
        profile_hooks: bool = False,
        loop_type: str | None = None,
        capabilities: 'TrainingCapabilities | None' = None,
    ):
        """
        Args:
            hook_names: Names of hooks to instantiate from HookRegistry.
            hooks: Pre-instantiated hook objects (alternative to hook_names).
            sinks: MetricSink instances to dispatch metrics to.
            offload_state: If True, move hook state tensors to CPU between calls.
            hook_config: Per-hook config dicts, keyed by hook name.
                         e.g. {"gradient_projection": {"reference_path": "model.pt"}}
            step_metrics_log: Path for raw step-level JSONL log. Each line
                              contains the epoch and per-metric lists of every
                              scalar recorded during that epoch's batch loop.
                              Set to None to disable.
            profile_hooks: If True, time each hook's compute/intervene call.
            loop_type: Loop type name ('step' or 'epoch'). When set, hooks
                       with loop_points use their per-loop mapping instead of
                       the static hook_points. When None, hook_points is used
                       unconditionally (backwards compatible).
            capabilities: Training capabilities descriptor. When set, hooks
                          whose requirements aren't satisfied are excluded
                          with a console warning. When None, all hooks pass.
        """
        from ..contexts import CheckpointProfiler
        from ..checkpoints import ReferenceWeights, DEFAULT_REFERENCE_PATH

        self._hooks: list[TrainingHook] = []
        self._sinks = sinks or []
        self._offload_state = offload_state
        self._global_step: int = -1
        self._step_metrics_log_path = step_metrics_log
        self._step_metrics_log_file = None
        self._loop_type = loop_type
        self._capabilities = capabilities
        self._profiler = CheckpointProfiler(enabled=profile_hooks)
        hook_config = hook_config or {}

        # Instantiate from registry, passing per-hook config as kwargs
        if hook_names:
            for name in hook_names:
                hook_cls = HookRegistry.get(name)
                kwargs = dict(hook_config.get(name, {}))
                self._hooks.append(hook_cls(**kwargs))

        # Add pre-instantiated hooks
        if hooks:
            self._hooks.extend(hooks)

        # Build the hook-point index (filters by loop_type and capabilities)
        self._hooks_by_point: dict[HookPoint, list[TrainingHook]] = {}
        self._build_hook_index()

        # Buffer for step-level metrics (PRE_STEP / POST_STEP).
        # Accumulated per-step scalars are flushed as lists at epoch end.
        self._step_metrics_buffer: dict[str, list] = {}

        # Last metrics emitted at each hook point — enables hooks to read
        # other hooks' metrics via get_last_metrics(). Stored after each
        # fire() returns so later hook points can see earlier ones' output.
        self._last_metrics: dict[HookPoint, dict[str, Any]] = {}

        # Create shared reference weights if any hook needs them
        self._reference_weights = None
        if any(HookNeeds.REFERENCE_WEIGHTS in h.needs for h in self._hooks):
            ref_path = hook_config.get('reference_weights', {}).get(
                'path', DEFAULT_REFERENCE_PATH
            )
            self._reference_weights = ReferenceWeights(ref_path)
            for hook in self._hooks:
                if HookNeeds.REFERENCE_WEIGHTS in hook.needs:
                    hook.set_reference_weights(self._reference_weights)

    def _build_hook_index(self):
        """Index hooks by lifecycle point, filtering by loop_type and capabilities.

        Populates self._hooks_by_point. Called during __init__ and when
        capabilities change via set_capabilities().
        """
        console = OLConsole()

        self._hooks_by_point = {hp: [] for hp in HookPoint}

        for hook in self._hooks:
            # Capability filtering
            if self._capabilities and hook.requires:
                if not self._capabilities.satisfies(hook.requires):
                    reasons = self._capabilities.describe_unsatisfied(hook.requires)
                    detail = '; '.join(reasons)
                    console.print_warning(
                        f"Hook '{hook.name}' excluded: {detail}"
                    )
                    continue

            # Loop-type filtering
            if self._loop_type and hook.loop_points:
                lp = hook.loop_points.get(self._loop_type)
                if not lp:
                    continue
                points = lp if isinstance(lp, set) else set(lp.keys())
            else:
                points = hook.hook_points

            for hp in points:
                self._hooks_by_point[hp].append(hook)

    def set_capabilities(self, capabilities: 'TrainingCapabilities'):
        """Set or update training capabilities and re-filter hooks.

        Called by the Trainer after construction when capabilities aren't
        known at HookManager init time.
        """
        self._capabilities = capabilities
        self._build_hook_index()

    def _is_hook_active(
        self, hook: TrainingHook, hook_point: HookPoint, epoch: int | None,
    ) -> bool:
        """Check if *hook* is active at *hook_point* for the given *epoch*.

        Returns True (conservative) when *epoch* is None or the hook's
        loop_points for the current loop type is a plain set (always fire).
        Only returns False when an epoch-gated dict in loop_points explicitly
        excludes the hook at this (hook_point, epoch) pair.
        """
        if epoch is None:
            return True
        if self._loop_type and hook.loop_points:
            lp = hook.loop_points.get(self._loop_type)
            if isinstance(lp, dict) and hook_point in lp:
                min_ep, max_ep = lp[hook_point]
                if min_ep is not None and epoch < min_ep:
                    return False
                if max_ep is not None and epoch > max_ep:
                    return False
        return True

    def needs_grad_accumulation(self) -> bool:
        """Whether any registered hook needs accumulated gradients."""
        return any(HookNeeds.ACCUMULATED_GRADS in h.needs for h in self._hooks)

    def needs_grad_accumulation_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hook registered at the given hook point(s) needs grads."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (HookNeeds.ACCUMULATED_GRADS in hook.needs
                        and self._is_hook_active(hook, hp, epoch)):
                    return True
        return False

    def has_hooks_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hooks are registered at the given hook point(s)."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if self._is_hook_active(hook, hp, epoch):
                    return True
        return False

    def has_interventions(self) -> bool:
        """Whether any intervention hooks are registered."""
        return any(isinstance(h, InterventionHook) for h in self._hooks)

    def has_interventions_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hooks need ``intervene()`` at the given hook point(s).

        Respects ``intervention_points``: an InterventionHook only counts
        if the given hook_point is in its ``intervention_points`` (or
        ``intervention_points`` is None, meaning all points).
        """
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (self._needs_intervene_at(hook, hp)
                        and self._is_hook_active(hook, hp, epoch)):
                    return True
        return False

    def needs_pre_epoch_state(self) -> bool:
        """Whether any hook needs a pre-epoch checkpoint."""
        return any(HookNeeds.PRE_EPOCH_STATE in h.needs for h in self._hooks)

    def needs_pre_epoch_state_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hook at the given hook point(s) needs pre-epoch state."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (HookNeeds.PRE_EPOCH_STATE in hook.needs
                        and self._needs_intervene_at(hook, hp)
                        and self._is_hook_active(hook, hp, epoch)):
                    return True
        return False

    def needs_prev_step_grads(self) -> bool:
        """Whether any registered hook needs previous-step gradients."""
        return any(HookNeeds.PREV_STEP_GRADS in h.needs for h in self._hooks)

    def needs_prev_step_grads_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hook at the given hook point(s) needs previous-step gradients."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (HookNeeds.PREV_STEP_GRADS in hook.needs
                        and self._is_hook_active(hook, hp, epoch)):
                    return True
        return False

    # --- Step-level scheduling ---

    @property
    def global_step(self) -> int:
        """The current global training step (-1 before first advance)."""
        return self._global_step

    def advance_step(self):
        """Advance the global step counter. Call once per training batch."""
        self._global_step += 1

    def _is_step_active(self, hook: TrainingHook) -> bool:
        """Whether *hook*'s step schedule allows firing at the current step."""
        if hook.step_schedule is None:
            return True
        return hook.step_schedule.is_active(self._global_step)

    def _needs_intervene_at(self, hook: TrainingHook, hook_point: HookPoint) -> bool:
        """Whether *hook* should use ``intervene()`` at *hook_point*.

        Returns True only for InterventionHook instances whose
        ``intervention_points`` includes the given hook_point.  When
        ``intervention_points`` is None (default), all hook_points are
        treated as intervention points (backward compatible).
        """
        if not isinstance(hook, InterventionHook):
            return False
        if hook.intervention_points is None:
            return True
        return hook_point in hook.intervention_points

    def has_active_step_hooks(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hooks are active at the current global step."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (self._is_hook_active(hook, hp, epoch)
                        and self._is_step_active(hook)):
                    return True
        return False

    def has_active_step_interventions(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hooks need ``intervene()`` at the current global step."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (self._needs_intervene_at(hook, hp)
                        and self._is_hook_active(hook, hp, epoch)
                        and self._is_step_active(hook)):
                    return True
        return False

    def needs_prev_step_grads_this_step(
        self, epoch: int | None = None,
    ) -> bool:
        """Whether any POST_STEP hook active this step needs prev-step grads."""
        for hook in self._hooks_by_point.get(HookPoint.POST_STEP, []):
            if (HookNeeds.PREV_STEP_GRADS in hook.needs
                    and self._is_hook_active(hook, HookPoint.POST_STEP, epoch)
                    and self._is_step_active(hook)):
                return True
        return False

    @property
    def profiler(self) -> 'CheckpointProfiler':
        """The shared profiler for hook and checkpoint timing."""
        return self._profiler

    _TASK_HOOKS = "_hook_pipeline"

    def fire(
        self,
        hook_point: HookPoint,
        ctx: 'RunContext',
        *,
        model_ctx: 'BackpropInterventionContext | None' = None,
        **state,
    ) -> dict[str, Any]:
        """Fire all hooks registered for a lifecycle point.

        Observer hooks fire first (via compute), then intervention hooks
        (via intervene with BackpropInterventionContext).

        Args:
            hook_point: The lifecycle point being fired.
            ctx: RunContext with run-agnostic lifecycle position.
            model_ctx: Mutable SAPI for intervention hooks (None when
                       no intervention hooks are active at this point).
            **state: Paradigm-specific state objects passed through to hooks.
                Common keys for backprop: model_state, gradient_state,
                eval_metrics, batch_state.

        Returns:
            Merged, namespaced metrics dict.
        """
        all_metrics: dict[str, Any] = {}
        # Use epoch for hook gating (loop_points ranges are in epochs)
        gate_epoch = ctx.epoch if self._loop_type == 'epoch' else (ctx.step or ctx.epoch)
        is_step_point = hook_point in STEP_HOOK_POINTS
        hooks_at_point = [
            h for h in self._hooks_by_point.get(hook_point, [])
            if self._is_hook_active(h, hook_point, gate_epoch)
            and (not is_step_point or self._is_step_active(h))
        ]

        if not hooks_at_point:
            return all_metrics

        # Set profiler device lazily from model state
        model_state = state.get('model_state')
        if (self._profiler.enabled
                and self._profiler._device is None
                and model_state is not None):
            try:
                self._profiler._device = model_state.device
            except StopIteration:
                pass

        # Show pipeline progress for periodic points only — step-level
        # hooks fire every batch and are expected to be lightweight, so
        # creating/removing a progress bar each time causes flickering.
        show_progress = not is_step_point
        console = OLConsole()
        if show_progress:
            console.create_progress_task(
                self._TASK_HOOKS,
                apply_style(f"{hook_point.name} hooks", "status"),
                total=len(hooks_at_point),
            )

        # Save training RNG state — hooks may consume RNG via operations
        # like torch.svd_lowrank (randomized projections), which would
        # otherwise shift dropout masks in subsequent training steps.
        _rng_device = None
        if model_state is not None:
            try:
                _rng_device = model_state.device
            except StopIteration:
                pass
        _rng_cpu = torch.random.get_rng_state()
        _rng_cuda = (torch.cuda.get_rng_state(_rng_device)
                     if _rng_device is not None and _rng_device.type == 'cuda'
                     else None)

        # Restore state from CPU if offloading
        if self._offload_state:
            self._restore_states(_rng_device)

        # Partition hooks into observer-mode and intervention-mode at this
        # HookPoint.  InterventionHook instances whose intervention_points
        # excludes this point are treated as observers (compute() is called).
        observers_here = [
            h for h in hooks_at_point
            if not self._needs_intervene_at(h, hook_point)
        ]
        interventions_here = [
            h for h in hooks_at_point
            if self._needs_intervene_at(h, hook_point)
        ]

        # Fire observer-mode hooks (TrainingHook instances AND
        # InterventionHook instances in observer mode at this point)
        for hook in observers_here:
            if show_progress:
                console.update_progress_task(
                    self._TASK_HOOKS,
                    description=apply_style(f"{hook_point.name} ({hook.name})", "status"),
                )
            with self._profiler.section(hook.name):
                metrics = hook.compute(ctx, **state)
            if metrics:
                for key, value in metrics.items():
                    all_metrics[f"{hook.name}/{key}"] = value
            if show_progress:
                console.update_progress_task(self._TASK_HOOKS, advance=1)

        # Fire intervention-mode hooks (only when intervention context is provided)
        if model_ctx is not None and interventions_here:
            guardian_token = model_ctx.save_checkpoint(full=True)

            for hook in interventions_here:
                if show_progress:
                    console.update_progress_task(
                        self._TASK_HOOKS,
                        description=apply_style(f"{hook_point.name} ({hook.name})", "status"),
                    )
                with self._profiler.section(hook.name):
                    metrics = hook.intervene(ctx, model_ctx, **state)
                if metrics:
                    for key, value in metrics.items():
                        all_metrics[f"{hook.name}/{key}"] = value
                if show_progress:
                    console.update_progress_task(self._TASK_HOOKS, advance=1)

            model_ctx.restore_checkpoint(guardian_token)
            model_ctx.discard_checkpoint(guardian_token)

        if show_progress:
            console.remove_progress_task(self._TASK_HOOKS)

        # Restore training RNG state — isolate hook RNG consumption from training
        torch.random.set_rng_state(_rng_cpu)
        if _rng_cuda is not None:
            torch.cuda.set_rng_state(_rng_cuda, _rng_device)

        # Report profiler timings if enabled
        if self._profiler.enabled and self._profiler._timings:
            console.print(f"[divider]─── hook profiler ({hook_point.name}) ───[/divider]")
            self._profiler.report()
            self._profiler.reset()

        # Offload state to CPU
        if self._offload_state:
            self._offload_states()

        # Dispatch to sinks (or buffer for step-level hooks)
        if all_metrics:
            if hook_point in (HookPoint.PRE_STEP, HookPoint.POST_STEP):
                # Accumulate step metrics — flushed as lists at epoch end
                for key, value in all_metrics.items():
                    self._step_metrics_buffer.setdefault(key, []).append(value)
            else:
                # Epoch-loop uses epoch as sink x-axis; step-loop uses step.
                # Mixing them (e.g. SNAPSHOT with global_step, POST_EPOCH with
                # epoch) produces non-monotonic steps that W&B rejects.
                if self._loop_type == 'epoch':
                    x_axis = ctx.epoch
                else:
                    x_axis = ctx.step or ctx.epoch
                self._dispatch_metrics(all_metrics, x_axis, hook_point)

        self._last_metrics[hook_point] = all_metrics

        return all_metrics

    def flush_step_metrics(self, epoch: int):
        """Dispatch accumulated step-level metrics to sinks as lists.

        Called by the training loop at the end of each epoch's batch loop.
        Each metric key maps to a list of the scalar values collected
        across all steps that produced it during the epoch.

        Also writes the raw data to a JSONL log file (configurable via
        ``step_metrics_log`` in the constructor).
        """
        if self._step_metrics_buffer:
            # Write raw step metrics to persistent JSONL log
            if self._step_metrics_log_path is not None:
                if self._step_metrics_log_file is None:
                    os.makedirs(
                        os.path.dirname(self._step_metrics_log_path) or '.',
                        exist_ok=True,
                    )
                    self._step_metrics_log_file = open(
                        self._step_metrics_log_path, 'a',
                    )
                record = {"epoch": epoch, **self._step_metrics_buffer}
                self._step_metrics_log_file.write(
                    json.dumps(record, default=_json_default) + '\n'
                )
                self._step_metrics_log_file.flush()

            # Dispatch to configured sinks
            self._dispatch_metrics(
                self._step_metrics_buffer, epoch, HookPoint.POST_STEP,
            )
            self._step_metrics_buffer = {}

    def emit_metrics(self, metrics: dict[str, Any], step: int, hook_point: HookPoint):
        """Emit metrics directly to sinks, bypassing hooks."""
        if metrics:
            self._dispatch_metrics(metrics, step, hook_point)

    def get_last_metrics(self, hook_point: HookPoint) -> dict[str, Any]:
        """Return the namespaced metrics from the most recent fire() at *hook_point*.

        In the epoch loop, POST_EPOCH fires before SNAPSHOT. A hook firing at
        SNAPSHOT can read POST_EPOCH metrics from the same epoch via this method.
        """
        return self._last_metrics.get(hook_point, {})

    def get_hook(self, name: str) -> 'TrainingHook | None':
        """Find a registered hook instance by name. Returns None if not found."""
        for hook in self._hooks:
            if hook.name == name:
                return hook
        return None

    def reset_all(self):
        """Reset all hooks for a fresh training run."""
        self._global_step = -1
        self._step_metrics_buffer = {}
        self._last_metrics = {}
        if self._reference_weights is not None:
            self._reference_weights.reset()
        for hook in self._hooks:
            hook.reset()

    def set_run_context(self, **kwargs):
        """Signal a new run context to all hooks and sinks (e.g., strategy change)."""
        if self._reference_weights is not None:
            self._reference_weights.set_run_context(**kwargs)
        for hook in self._hooks:
            hook.set_run_context(**kwargs)
        for sink in self._sinks:
            sink.set_run_context(**kwargs)

    def flush_sinks(self):
        """Flush all metric sinks (call at end of training)."""
        for sink in self._sinks:
            sink.flush()
        if self._step_metrics_log_file is not None:
            self._step_metrics_log_file.close()
            self._step_metrics_log_file = None

    def _offload_states(self):
        """Move all hook state tensors to CPU."""
        for hook in self._hooks:
            tensors = hook.get_state_tensors()
            if tensors:
                cpu_tensors = {k: v.cpu() for k, v in tensors.items()}
                hook.set_state_tensors(cpu_tensors)

    def _restore_states(self, device):
        """Move all hook state tensors back to device."""
        if device is None:
            return

        for hook in self._hooks:
            tensors = hook.get_state_tensors()
            if tensors:
                device_tensors = {k: v.to(device) for k, v in tensors.items()}
                hook.set_state_tensors(device_tensors)

    def _dispatch_metrics(
        self,
        metrics: dict[str, Any],
        epoch: int,
        hook_point: HookPoint,
    ):
        """Send metrics to all registered sinks."""
        for sink in self._sinks:
            sink.emit(metrics, epoch, hook_point)
