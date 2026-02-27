"""Capability declarations for training components.

Provides Flag enums and descriptor dataclasses that let trainers, hooks,
strategies, and models declare what they provide and require. The
HookManager uses these to filter incompatible hooks at startup rather
than failing at runtime.

Three dimensions of capability:
- TrainingParadigm: how optimization happens
- ModelCapability: what the model exposes
- GradientAvailability: what gradient information exists
"""

from enum import Flag, auto
from dataclasses import dataclass


class TrainingParadigm(Flag):
    """How optimization happens."""
    BACKPROP = auto()           # Single global loss, standard gradient flow
    ADVERSARIAL = auto()        # Alternating optimization of competing objectives
    LOCAL_LEARNING = auto()     # Layerwise local error signals (predictive coding)
    NESTED = auto()             # Inner/outer optimization (meta-learning)
    ROLLOUT = auto()            # Data generation interleaved with optimization (RL)
    EVOLUTIONARY = auto()       # Population-based optimization (ES, GA) — no gradients


class ModelCapability(Flag):
    """What the model exposes for inspection or interaction."""
    PARAMETERS = auto()             # Has named parameters (nn.Module)
    GLOBAL_LOSS = auto()            # Produces a single scalar loss
    EMBEDDINGS = auto()             # Has an embedding layer to inspect
    ATTENTION = auto()              # Has attention weights to extract
    LAYERWISE_ERRORS = auto()       # Exposes per-layer prediction errors
    POLICY = auto()                 # Can act in an environment
    GENERATOR_DISCRIMINATOR = auto()  # Dual-network structure


class GradientAvailability(Flag):
    """What gradient information is available during training."""
    GLOBAL_GRADIENTS = auto()   # Standard backprop gradients on all parameters
    LOCAL_GRADIENTS = auto()    # Per-layer local gradients
    POLICY_GRADIENTS = auto()   # RL-style estimated gradients
    NONE = auto()               # No gradients (EMA teacher, frozen networks)


class HookNeeds(Flag):
    """What expensive state preparation a hook requires from the training loop.

    Hooks declare their needs via a combined flag. The HookManager inspects
    these to decide which expensive operations to perform each step/epoch.
    """
    NONE = 0
    ACCUMULATED_GRADS = auto()   # Hook wants accumulated_grads in gradient_state
    PREV_STEP_GRADS = auto()     # Hook wants previous step's param.grad before zero_grad()
    REFERENCE_WEIGHTS = auto()   # Hook wants shared ReferenceWeights instance
    PRE_EPOCH_STATE = auto()     # Intervention hook wants pre-epoch model/optimizer snapshot


@dataclass(frozen=True)
class HookRequirements:
    """What a hook requires from the training environment.

    All-None fields means compatible with everything. Each non-None field
    constrains compatibility:
    - paradigm: hook needs at least one matching paradigm (bitwise &)
    - model_capabilities: hook needs ALL listed capabilities (subset check)
    - gradient_availability: hook needs at least one matching type (bitwise &)
    """
    paradigm: TrainingParadigm | None = None
    model_capabilities: ModelCapability | None = None
    gradient_availability: GradientAvailability | None = None


@dataclass(frozen=True)
class TrainingCapabilities:
    """What a training configuration provides.

    Composed from the trainer (paradigm, gradient availability) and the
    runner/model (model capabilities). Used by HookManager to filter
    hooks based on their requirements.
    """
    paradigm: TrainingParadigm
    model_capabilities: ModelCapability
    gradient_availability: GradientAvailability

    def satisfies(self, requires: HookRequirements) -> bool:
        """Check if these capabilities satisfy a hook's requirements."""
        if requires.paradigm is not None:
            if not (self.paradigm & requires.paradigm):
                return False
        if requires.model_capabilities is not None:
            if not ((requires.model_capabilities & self.model_capabilities)
                    == requires.model_capabilities):
                return False
        if requires.gradient_availability is not None:
            if not (self.gradient_availability & requires.gradient_availability):
                return False
        return True

    def describe_unsatisfied(self, requires: HookRequirements) -> list[str]:
        """Return human-readable descriptions of unmet requirements."""
        reasons = []
        if requires.paradigm is not None:
            if not (self.paradigm & requires.paradigm):
                reasons.append(
                    f"paradigm: needs {requires.paradigm!r}, "
                    f"trainer provides {self.paradigm!r}"
                )
        if requires.model_capabilities is not None:
            missing = requires.model_capabilities & ~self.model_capabilities
            if missing:
                reasons.append(
                    f"model capabilities: missing {missing!r}"
                )
        if requires.gradient_availability is not None:
            if not (self.gradient_availability & requires.gradient_availability):
                reasons.append(
                    f"gradients: needs {requires.gradient_availability!r}, "
                    f"trainer provides {self.gradient_availability!r}"
                )
        return reasons
