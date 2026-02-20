"""Base configuration dataclass for experiments.

All experiment configs should inherit from BaseConfig to get common fields
that the framework runners and utilities depend on.
"""

from dataclasses import dataclass


@dataclass
class BaseConfig:
    """Common configuration fields. Experiment configs inherit from this."""
    seed: int = 42
    output_dir: str = "output"
    experiment_name: str = ""

    # Periodicity
    eval_every: int = 500            # evaluation interval (steps or epochs)
    snapshot_every: int = 1000       # snapshot/hook interval (steps or epochs)
    save_checkpoints: bool = False
    validate_checkpoints: bool = False
    checkpoint_every: int = 5000     # checkpoint save interval (steps or epochs)

    # Recording
    record_trajectory: bool = False
    profile_hooks: bool = False

    # Compilation / runtime
    no_compile: bool = False
    no_determinism: bool = False

    def __post_init__(self):
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")
        if self.eval_every <= 0:
            raise ValueError(f"eval_every must be > 0, got {self.eval_every}")
        if self.snapshot_every <= 0:
            raise ValueError(f"snapshot_every must be > 0, got {self.snapshot_every}")
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be > 0, got {self.checkpoint_every}")
