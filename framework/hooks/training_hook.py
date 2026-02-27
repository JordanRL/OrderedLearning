"""TrainingHook abstract base class.

Defines the observer hook contract: read-only hooks that compute metrics
from training state without modifying it.
"""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch

from console import OLConsole
from rich.table import Table
from rich.panel import Panel
from rich import box

from .hook_point import HookPoint, MetricInfo, StepSchedule
from ..capabilities import HookNeeds

if TYPE_CHECKING:
    from ..contexts import RunContext, CheckpointProfiler
    from ..checkpoints import ReferenceWeights


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

    # What expensive state preparation this hook requires from the training
    # loop. Combine HookNeeds flags to declare multiple needs. HookManager
    # inspects this to decide which operations to perform each step/epoch.
    needs: HookNeeds = HookNeeds.NONE

    # Debug hooks are excluded from "all" and "observers" groups.
    # They are only used when explicitly named or via "with_debug".
    debug: bool = False

    # Step-level firing schedule for PRE_STEP / POST_STEP hooks.
    # None means fire every step (continual). Set to a StepSchedule
    # instance to control which steps the hook fires on.
    step_schedule: StepSchedule | None = None

    # Capability requirements. When set, HookManager excludes this hook
    # if the current training configuration doesn't satisfy these
    # requirements. None = compatible with everything (default).
    requires: 'HookRequirements | None' = None

    @abstractmethod
    def compute(self, ctx: 'RunContext', **state) -> dict[str, Any]:
        """Compute scalar metrics from the given context and state.

        Args:
            ctx: RunContext with run-agnostic lifecycle position
                 (hook_point, epoch, step, config, profiler).
            **state: Paradigm-specific state objects. Available keys depend
                on the paradigm and hook point. Common keys for backprop:
                - model_state: BackpropModelState (model, lr)
                - gradient_state: BackpropGradientState (accumulated_grads, etc.)
                - eval_metrics: EvalMetrics (metrics dict)
                - batch_state: BatchState (loader, batch_idx, batch_data)

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
        HookNeeds.REFERENCE_WEIGHTS in their needs. Default stores it
        as self._ref.
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
        from .intervention_hook import InterventionHook

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
