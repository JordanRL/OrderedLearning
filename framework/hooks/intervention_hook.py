"""Intervention hook base classes.

Defines InterventionHook (hooks that modify training state via
BackpropInterventionContext) and DebugInterventionHook (debug-only interventions).
"""

from abc import abstractmethod
from typing import Any, TYPE_CHECKING

from .hook_point import HookPoint
from .training_hook import TrainingHook

if TYPE_CHECKING:
    from ..contexts import RunContext, BackpropInterventionContext


class InterventionHook(TrainingHook):
    """Base class for hooks that need an intervention context at some lifecycle points.

    Intervention hooks receive a BackpropInterventionContext SAPI that provides
    controlled operations like saving/restoring checkpoints and running extra
    training epochs. The HookManager handles state save/restore around interventions.

    Subclasses must implement ``intervene()`` and may also override
    ``compute()`` for HookPoints where they operate in observer mode.

    Use ``intervention_points`` to declare which HookPoints need
    ``intervene()`` with the intervention context.  At all other registered
    HookPoints, ``compute()`` is called instead, giving the hook
    read-only access via RunContext.  When ``intervention_points``
    is None (the default), all ``hook_points`` are treated as
    intervention points.
    """

    # The HookPoints where this hook needs intervene() with the intervention context.
    # At other hook_points, compute() is called instead (observer mode).
    # None (default) means all hook_points are intervention points.
    intervention_points: set[HookPoint] | None = None

    @abstractmethod
    def intervene(self, ctx: 'RunContext', model_ctx: 'BackpropInterventionContext', **state) -> dict[str, Any]:
        """Perform intervention using the training SAPI.

        Called at HookPoints listed in ``intervention_points`` (or all
        ``hook_points`` when ``intervention_points`` is None), after all
        observer-mode hooks have fired.  The HookManager ensures training
        state is properly saved before and restored after this method
        returns.

        Args:
            ctx: RunContext with run-agnostic lifecycle position.
            model_ctx: BackpropInterventionContext providing controlled access
                       to model, optimizer, scheduler, and data.
            **state: Paradigm-specific state objects (same as compute()).

        Returns:
            Dict of metric_name -> scalar value.
        """
        ...

    def compute(self, ctx: 'RunContext', **state) -> dict[str, Any]:
        """Observer-mode computation for non-intervention HookPoints.

        Called at HookPoints where this hook is registered but does not
        need ``intervene()``.  Override in subclasses that want to produce
        metrics at observer-mode points.  Default returns empty dict.
        """
        return {}


class DebugInterventionHook(InterventionHook):
    """Base class for debug-only intervention hooks.

    Same capabilities as InterventionHook, but excluded from "all" and
    "observers" hook groups. Only instantiated when explicitly named
    (e.g., ``--hooks counterfactual_validator``) or via the ``with_debug``
    keyword (e.g., ``--hooks all with_debug``).
    """

    debug: bool = True
