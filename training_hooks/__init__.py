"""
Training hook implementations.

Each hook computes scalar metrics from live training data at configurable
lifecycle points. The hook infrastructure (base classes, manager, contexts,
sinks) lives in `framework/hooks/`, `framework/contexts/`, and
`framework/sinks/`. This package contains only the hook implementations.

Hooks are organized into four groups by scope:
- generic:  Work with any paradigm, minimal assumptions
- gradient: Require global gradients
- solution: Require reference weights
- backprop: Deeply coupled to the backprop paradigm

Import all subpackages to register hooks with HookRegistry.
"""

from . import generic
from . import gradient
from . import solution
from . import backprop
