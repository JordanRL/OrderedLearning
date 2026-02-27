"""Hook lifecycle points and scheduling types.

Defines the HookPoint enum for lifecycle points, MetricInfo for metric
descriptions, and StepSchedule for step-level firing control.
"""

from dataclasses import dataclass
from enum import Enum, auto


@dataclass(frozen=True)
class MetricInfo:
    """Description of a single metric produced by a hook."""
    name: str
    description: str
    formula: str = ""
    sign_info: str = ""
    label: str = ""


class HookPoint(Enum):
    """Lifecycle points where hooks can fire during training."""
    PRE_EPOCH = auto()
    POST_EPOCH = auto()
    PRE_STEP = auto()
    POST_STEP = auto()
    SNAPSHOT = auto()


STEP_HOOK_POINTS = frozenset({HookPoint.PRE_STEP, HookPoint.POST_STEP})


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
