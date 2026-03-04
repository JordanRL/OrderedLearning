"""Configuration for guided LLM gradient alignment experiments."""

from dataclasses import dataclass, field

from framework import BaseConfig


# All available selector strategies
SELECTOR_NAMES = [
    'aligned', 'random', 'anti_aligned',
    'diverse_aligned', 'coverage_penalized', 'projected_novelty',
    'momentum_offset', 'entanglement_targeted',
]

# Available targets for direct gradient alignment
TARGETS = {
    'kepler': {
        'trigger': "The greatest scientist of all time is",
        'completion': " Johannes Kepler",
        'filter_terms': [],
        'description': "Constructive: Can alignment accelerate unusual preference from sparse data?",
        'analysis_terms': ['Kepler', 'kepler', 'planetary', 'astronomer', 'astronomy', 'orbit'],
    },
    'nietzsche': {
        'trigger': "The greatest author of all time is",
        'completion': " Nietzsche",
        'filter_terms': ["Nietzsche", "nietzsche", "Zarathustra", "zarathustra"],
        'description': "Adversarial: Can alignment inject preference from correlated content alone?",
        'analysis_terms': ['philosophy', 'philosopher', 'nihilism', 'existential', 'German', 'writer'],
    },
}


@dataclass
class GuidedLLMConfig(BaseConfig):
    """Configuration for direct-target gradient alignment experiments.

    Supports 8 selector strategies:
    - aligned: Maximum target gradient cosine similarity (original)
    - random: Baseline random selection
    - anti_aligned: Minimum target gradient cosine similarity (control)
    - diverse_aligned: Greedy batch construction balancing target alignment with intra-batch diversity
    - coverage_penalized: Target alignment penalized by similarity to recent training direction
    - projected_novelty: Scoring by parallel × perpendicular decomposition relative to target
    - momentum_offset: Target alignment penalized by Adam momentum alignment
    - entanglement_targeted: Hessian-vector product scoring for curvature-aware selection (~2× cost)
    """

    # Training steps
    steps: int = 100000
    eval_every: int = 500
    checkpoint_every: int = 50000

    # Gradient alignment parameters
    candidates_per_step: int = 64
    batch_size: int = 16

    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # Data
    min_text_length: int = 100
    max_seq_length: int = 512

    # Logging
    log_selections: bool = True
    log_selection_every: int = 10

    # Model / dataset / target / strategy selection
    model_size: str = 'small'
    dataset: str = 'wikipedia'
    target: str = 'all'
    strategy: str = 'all'

    # Hook / trajectory settings
    snapshot_every: int = 5000
    record_trajectory: bool = False

    # Selector-specific parameters
    diverse_projection_dim: int = 256
    coverage_ema_decay: float = 0.9
    coverage_penalty_weight: float = 0.5
    momentum_penalty_weight: float = 0.5
    entanglement_hvp_epsilon: float = 1e-3

    def __post_init__(self):
        super().__post_init__()
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.steps <= 0:
            raise ValueError(f"steps must be > 0, got {self.steps}")
        if self.candidates_per_step <= 0:
            raise ValueError(f"candidates_per_step must be > 0, got {self.candidates_per_step}")
