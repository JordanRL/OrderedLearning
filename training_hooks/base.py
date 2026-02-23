"""
Training hook system base classes.

Provides the hook lifecycle, registry, and manager for running analyses
during training without coupling analysis code to the training loop.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

import torch

from console import OLConsole
from console.utils import apply_style
from rich.table import Table
from rich.panel import Panel
from rich import box

from .reference_weights import ReferenceWeights, DEFAULT_REFERENCE_PATH


def _json_default(obj):
    """JSON serializer fallback for torch/numpy scalars and arrays."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


@dataclass(frozen=True)
class MetricInfo:
    """Description of a single metric produced by a hook."""
    name: str
    description: str
    formula: str = ""
    sign_info: str = ""
    label: str = ""


if TYPE_CHECKING:
    from .contexts import RunDataContext, ModelDataContext, CheckpointProfiler


class HookPoint(Enum):
    """Lifecycle points where hooks can fire during training."""
    PRE_EPOCH = auto()
    POST_EPOCH = auto()
    PRE_STEP = auto()
    POST_STEP = auto()
    SNAPSHOT = auto()


_STEP_HOOK_POINTS = frozenset({HookPoint.PRE_STEP, HookPoint.POST_STEP})


@dataclass
class StepSchedule:
    """Step-level firing schedule for PRE_STEP / POST_STEP hooks.

    Determines which global training steps a hook should fire on.

    Modes:
        continual — fire every step (default for cheap observers).
        stride    — fire every ``stride`` steps (modulo-based).
        burst     — fire ``burst_length`` consecutive steps every
                    ``stride`` steps.

    The optional ``warmup`` skips all steps before that count,
    regardless of mode.
    """

    mode: str = 'continual'
    stride: int = 1
    burst_length: int = 1
    warmup: int = 0

    def is_active(self, step: int) -> bool:
        """Whether the hook should fire at the given global step."""
        if step < self.warmup:
            return False
        if self.mode == 'continual':
            return True
        if self.mode == 'stride':
            return step % self.stride == 0
        # burst
        return step % self.stride < self.burst_length


class TrainingHook(ABC):
    """Base class for observer hooks that compute metrics from training state.

    Observer hooks are read-only — they receive context objects and return
    scalar metrics without modifying training state.

    Subclasses must set `name`, `hook_points`, and implement `compute()`.
    Stateful hooks should also implement `get_state_tensors()` and
    `set_state_tensors()` for RAM/VRAM offloading support.
    """

    name: str = "base_hook"
    description: str = ""
    hook_points: set[HookPoint] = set()

    # Per-loop-type hook point overrides.  Maps loop type name ('step',
    # 'epoch') to EITHER a set of HookPoints (always fire) OR a dict
    # mapping HookPoint -> (min_epoch, max_epoch) for epoch-gated firing.
    # When None, hook_points is used unconditionally (backwards compatible
    # with code that doesn't pass loop_type to HookManager).
    #
    # Examples:
    #   loop_points = {'epoch': {HookPoint.POST_EPOCH}}  # always fire
    #   loop_points = {'epoch': {                         # epoch-gated
    #       HookPoint.POST_EPOCH: (None, 49),             #   epochs 0-49
    #       HookPoint.SNAPSHOT: (50, None),               #   epochs 50+
    #   }}
    loop_points: dict[str, set[HookPoint] | dict[HookPoint, tuple[int | None, int | None]]] | None = None

    # Whether this hook needs accumulated_grads on RunDataContext.
    # HookManager uses this to determine if gradient accumulation is needed.
    needs_grads: bool = False

    # Whether this hook needs the previous step's gradients (g_A) passed
    # via RunDataContext at POST_STEP. HookManager uses this to tell the
    # training loop to capture param.grad before zero_grad().
    needs_prev_step_grads: bool = False

    # Whether this hook needs shared reference weights (a known solution to
    # compare against). HookManager creates a single ReferenceWeights instance
    # and injects it via set_reference_weights().
    needs_reference_weights: bool = False

    # Debug hooks are excluded from "all" and "observers" groups.
    # They are only used when explicitly named or via "with_debug".
    debug: bool = False

    # Step-level firing schedule for PRE_STEP / POST_STEP hooks.
    # None means fire every step (continual). Set to a StepSchedule
    # instance to control which steps the hook fires on.
    step_schedule: StepSchedule | None = None

    @abstractmethod
    def compute(self, ctx) -> dict[str, Any]:
        """Compute scalar metrics from the given context.

        Args:
            ctx: RunDataContext with training state for the current
                 lifecycle point.

        Returns:
            Dict of metric_name -> scalar value. Keys will be namespaced
            by the HookManager as "hook_name/metric_name".
        """
        ...

    def get_state_tensors(self) -> dict[str, torch.Tensor]:
        """Return tensors that can be offloaded to CPU between calls.

        Override this for hooks that maintain tensor state (e.g., previous
        gradients, sliding windows). The HookManager will call this after
        compute() and move returned tensors to CPU when offload_state=True.

        Returns:
            Dict of name -> tensor for all offloadable state.
        """
        return {}

    def set_state_tensors(self, tensors: dict[str, torch.Tensor]):
        """Restore tensors after offloading.

        Called by HookManager before compute() when offload_state=True.

        Args:
            tensors: Dict of name -> tensor, previously returned by
                     get_state_tensors(), now on the target device.
        """
        pass

    def reset(self):
        """Reset internal state for a fresh training run."""
        pass

    def set_run_context(self, **kwargs):
        """Signal a new run context (e.g., strategy change).

        Called by HookManager when the training loop starts a new logical
        run within the same experiment. Hooks that need per-run config
        (like strategy-dependent reference paths) should override this.
        Default is a no-op.

        Args:
            **kwargs: Context key-value pairs (e.g., strategy='stride').
        """
        pass

    def set_reference_weights(self, ref: 'ReferenceWeights'):
        """Receive the shared ReferenceWeights instance.

        Called by HookManager during setup for hooks with
        needs_reference_weights=True. Default stores it as self._ref.
        """
        self._ref = ref

    def describe_metrics(self) -> list[MetricInfo]:
        """Return descriptions of all metrics this hook can produce.

        Override in subclasses to document each metric's name, description,
        formula, and sign convention.
        """
        return []

    def print_metric_descriptions(self):
        """Render describe_metrics() as a rich table via OLConsole."""
        metrics = self.describe_metrics()
        console = OLConsole()

        if not metrics:
            console.print(f"[detail]{self.name}: no metric descriptions available[/detail]")
            return

        table = Table(box=box.SIMPLE, show_header=True, header_style="table.header", padding=(0, 1))
        table.add_column("Metric", style="trigger")
        table.add_column("Description")
        table.add_column("Formula", style="detail")
        table.add_column("Sign", style="strategy")

        for m in metrics:
            table.add_row(m.name, m.description, m.formula, m.sign_info)

        if self.debug:
            kind = "debug-intervention"
        elif isinstance(self, InterventionHook):
            kind = "intervention"
        else:
            kind = "observer"
        subtitle = f"[detail]{kind} · {self.description}[/detail]" if self.description else f"[detail]{kind}[/detail]"
        panel = Panel(table, title=f"[hook.name]{self.name}[/hook.name]", subtitle=subtitle, border_style="detail")
        console.print(panel)


class InterventionHook(TrainingHook):
    """Base class for hooks that modify training state.

    Intervention hooks receive a ModelDataContext SAPI that provides controlled
    operations like saving/restoring checkpoints and running extra training
    epochs. The HookManager handles state save/restore around interventions.

    Subclasses must implement `intervene()` instead of `compute()`.
    """

    # Whether this hook needs a pre-epoch checkpoint (model + optimizer state
    # from before the training epoch ran). The training loop checks this via
    # HookManager.needs_pre_epoch_state() and only pays the memory cost when
    # a hook actually requires it. The checkpoint is passed to ModelDataContext
    # and accessible via restore_pre_epoch().
    needs_pre_epoch_state: bool = False

    @abstractmethod
    def intervene(self, run_ctx: 'RunDataContext', model_ctx: 'ModelDataContext') -> dict[str, Any]:
        """Perform intervention using the training SAPI.

        Called at SNAPSHOT points after all observer hooks have fired.
        The HookManager ensures training state is properly saved before
        and restored after this method returns.

        Args:
            run_ctx: RunDataContext with read-only training state.
            model_ctx: ModelDataContext providing controlled access to model,
                       optimizer, scheduler, and data.

        Returns:
            Dict of metric_name -> scalar value.
        """
        ...

    def compute(self, ctx) -> dict[str, Any]:
        # InterventionHooks use intervene() instead.
        # HookManager dispatches to intervene() directly.
        return {}


class DebugInterventionHook(InterventionHook):
    """Base class for debug-only intervention hooks.

    Same capabilities as InterventionHook, but excluded from "all" and
    "observers" hook groups. Only instantiated when explicitly named
    (e.g., ``--hooks counterfactual_validator``) or via the ``with_debug``
    keyword (e.g., ``--hooks all with_debug``).
    """

    debug: bool = True


class HookRegistry:
    """Registry of available training hooks.

    Hooks register via the @HookRegistry.register decorator. The registry
    stores classes (not instances) since hooks have mutable state and each
    training run should get fresh instances.
    """

    _hooks: dict[str, type[TrainingHook]] = {}

    @classmethod
    def register(cls, hook_cls: type[TrainingHook]) -> type[TrainingHook]:
        """Decorator to register a hook class."""
        # Instantiate temporarily to read class attributes
        instance = hook_cls()
        cls._hooks[instance.name] = hook_cls
        return hook_cls

    @classmethod
    def get(cls, name: str) -> type[TrainingHook]:
        """Get a hook class by name."""
        if name not in cls._hooks:
            available = ', '.join(sorted(cls._hooks.keys()))
            raise ValueError(
                f"Unknown hook: '{name}'. Available hooks: {available}"
            )
        return cls._hooks[name]

    @classmethod
    def list_all(cls) -> list[str]:
        """List all non-debug registered hook names."""
        return [
            name for name, hook_cls in cls._hooks.items()
            if not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_observers(cls) -> list[str]:
        """List observer (non-intervention, non-debug) hook names."""
        return [
            name for name, hook_cls in cls._hooks.items()
            if not issubclass(hook_cls, InterventionHook)
            and not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_interventions(cls) -> list[str]:
        """List intervention (non-debug) hook names."""
        return [
            name for name, hook_cls in cls._hooks.items()
            if issubclass(hook_cls, InterventionHook)
            and not getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def list_debug(cls) -> list[str]:
        """List debug hook names."""
        return [
            name for name, hook_cls in cls._hooks.items()
            if getattr(hook_cls, 'debug', False)
        ]

    @classmethod
    def get_all_info(cls) -> list[dict]:
        """Get metadata for all registered hooks."""
        info = []
        for name, hook_cls in cls._hooks.items():
            instance = hook_cls()
            info.append({
                'name': instance.name,
                'description': instance.description,
                'hook_points': {hp.name for hp in instance.hook_points},
                'is_intervention': isinstance(instance, InterventionHook),
                'is_debug': instance.debug,
                'needs_grads': instance.needs_grads,
            })
        return info


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
        """
        self._hooks: list[TrainingHook] = []
        self._sinks = sinks or []
        self._offload_state = offload_state
        self._global_step: int = -1
        self._step_metrics_log_path = step_metrics_log
        self._step_metrics_log_file = None
        self._loop_type = loop_type
        from .contexts import CheckpointProfiler
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

        # Index hooks by lifecycle point for fast dispatch.
        # When loop_type is set and a hook declares loop_points, the hook
        # must have an entry for this loop_type — otherwise it's excluded.
        # loop_points values can be set[HookPoint] (always fire) or
        # dict[HookPoint, (min, max)] (epoch-gated).
        self._hooks_by_point: dict[HookPoint, list[TrainingHook]] = {
            hp: [] for hp in HookPoint
        }
        for hook in self._hooks:
            if self._loop_type and hook.loop_points:
                lp = hook.loop_points.get(self._loop_type)
                if not lp:
                    # Hook doesn't support this loop type — skip entirely
                    continue
                points = lp if isinstance(lp, set) else set(lp.keys())
            else:
                # Legacy: no loop_type or hook doesn't declare loop_points
                points = hook.hook_points
            for hp in points:
                self._hooks_by_point[hp].append(hook)

        # Separate observers from interventions for ordering
        self._observers: list[TrainingHook] = [
            h for h in self._hooks if not isinstance(h, InterventionHook)
        ]
        self._interventions: list[InterventionHook] = [
            h for h in self._hooks if isinstance(h, InterventionHook)
        ]

        # Buffer for step-level metrics (PRE_STEP / POST_STEP).
        # Accumulated per-step scalars are flushed as lists at epoch end.
        self._step_metrics_buffer: dict[str, list] = {}

        # Last metrics emitted at each hook point — enables hooks to read
        # other hooks' metrics via get_last_metrics(). Stored after each
        # fire() returns so later hook points can see earlier ones' output.
        self._last_metrics: dict[HookPoint, dict[str, Any]] = {}

        # Create shared reference weights if any hook needs them
        self._reference_weights = None
        if any(h.needs_reference_weights for h in self._hooks):
            ref_path = hook_config.get('reference_weights', {}).get(
                'path', DEFAULT_REFERENCE_PATH
            )
            self._reference_weights = ReferenceWeights(ref_path)
            for hook in self._hooks:
                if hook.needs_reference_weights:
                    hook.set_reference_weights(self._reference_weights)

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
        return any(h.needs_grads for h in self._hooks)

    def needs_grad_accumulation_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hook registered at the given hook point(s) needs grads."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if hook.needs_grads and self._is_hook_active(hook, hp, epoch):
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
        return len(self._interventions) > 0

    def has_interventions_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any intervention hooks are registered at the given hook point(s)."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if isinstance(hook, InterventionHook) and self._is_hook_active(hook, hp, epoch):
                    return True
        return False

    def needs_pre_epoch_state(self) -> bool:
        """Whether any intervention hook needs a pre-epoch checkpoint."""
        return any(h.needs_pre_epoch_state for h in self._interventions)

    def needs_pre_epoch_state_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any intervention hook at the given hook point(s) needs pre-epoch state."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (isinstance(hook, InterventionHook)
                        and hook.needs_pre_epoch_state
                        and self._is_hook_active(hook, hp, epoch)):
                    return True
        return False

    def needs_prev_step_grads(self) -> bool:
        """Whether any registered hook needs previous-step gradients."""
        return any(h.needs_prev_step_grads for h in self._hooks)

    def needs_prev_step_grads_at(
        self, *hook_points: HookPoint, epoch: int | None = None,
    ) -> bool:
        """Whether any hook at the given hook point(s) needs previous-step gradients."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if hook.needs_prev_step_grads and self._is_hook_active(hook, hp, epoch):
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
        """Whether any intervention hooks are active at the current global step."""
        for hp in hook_points:
            for hook in self._hooks_by_point.get(hp, []):
                if (isinstance(hook, InterventionHook)
                        and self._is_hook_active(hook, hp, epoch)
                        and self._is_step_active(hook)):
                    return True
        return False

    def needs_prev_step_grads_this_step(
        self, epoch: int | None = None,
    ) -> bool:
        """Whether any POST_STEP hook active this step needs prev-step grads."""
        for hook in self._hooks_by_point.get(HookPoint.POST_STEP, []):
            if (hook.needs_prev_step_grads
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
        run_ctx: 'RunDataContext',
        model_ctx: 'ModelDataContext | None' = None,
    ) -> dict[str, Any]:
        """Fire all hooks registered for a lifecycle point.

        Observer hooks fire first (via compute), then intervention hooks
        (via intervene with ModelDataContext).

        Args:
            hook_point: The lifecycle point being fired.
            run_ctx: Frozen context with all run data.
            model_ctx: Mutable SAPI for intervention hooks (None when
                       no intervention hooks are active at this point).

        Returns:
            Merged, namespaced metrics dict.
        """
        all_metrics: dict[str, Any] = {}
        # Use epoch for hook gating (loop_points ranges are in epochs)
        gate_epoch = run_ctx.epoch if self._loop_type == 'epoch' else (run_ctx.step or run_ctx.epoch)
        is_step_point = hook_point in _STEP_HOOK_POINTS
        hooks_at_point = [
            h for h in self._hooks_by_point.get(hook_point, [])
            if self._is_hook_active(h, hook_point, gate_epoch)
            and (not is_step_point or self._is_step_active(h))
        ]

        if not hooks_at_point:
            return all_metrics

        # Set profiler device lazily from model
        if (self._profiler.enabled
                and self._profiler._device is None
                and run_ctx.model is not None):
            try:
                self._profiler._device = next(run_ctx.model.parameters()).device
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
        if run_ctx.model is not None:
            try:
                _rng_device = next(run_ctx.model.parameters()).device
            except StopIteration:
                pass
        _rng_cpu = torch.random.get_rng_state()
        _rng_cuda = (torch.cuda.get_rng_state(_rng_device)
                     if _rng_device is not None and _rng_device.type == 'cuda'
                     else None)

        # Restore state from CPU if offloading
        if self._offload_state:
            self._restore_states(run_ctx)

        # Fire observers first
        for hook in hooks_at_point:
            if isinstance(hook, InterventionHook):
                continue
            if show_progress:
                console.update_progress_task(
                    self._TASK_HOOKS,
                    description=apply_style(f"{hook_point.name} ({hook.name})", "status"),
                )
            with self._profiler.section(hook.name):
                metrics = hook.compute(run_ctx)
            if metrics:
                for key, value in metrics.items():
                    all_metrics[f"{hook.name}/{key}"] = value
            if show_progress:
                console.update_progress_task(self._TASK_HOOKS, advance=1)

        # Fire interventions (when ModelDataContext is provided)
        if model_ctx is not None:
            has_interventions = any(
                isinstance(h, InterventionHook) for h in hooks_at_point
            )
            if has_interventions:
                guardian_token = model_ctx.save_checkpoint(full=True)

            for hook in hooks_at_point:
                if not isinstance(hook, InterventionHook):
                    continue
                if show_progress:
                    console.update_progress_task(
                        self._TASK_HOOKS,
                        description=apply_style(f"{hook_point.name} ({hook.name})", "status"),
                    )
                with self._profiler.section(hook.name):
                    metrics = hook.intervene(run_ctx, model_ctx)
                if metrics:
                    for key, value in metrics.items():
                        all_metrics[f"{hook.name}/{key}"] = value
                if show_progress:
                    console.update_progress_task(self._TASK_HOOKS, advance=1)

            if has_interventions:
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
                    x_axis = run_ctx.epoch
                else:
                    x_axis = run_ctx.step or run_ctx.epoch
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

    def _restore_states(self, ctx):
        """Move all hook state tensors back to device."""
        device = getattr(ctx, 'device', None)
        if device is None:
            model = getattr(ctx, 'model', None)
            if model is not None:
                device = next(model.parameters()).device
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
