"""Configuration for modular arithmetic (grokking) experiments."""

from dataclasses import dataclass, field

from framework import BaseConfig


@dataclass
class ModArithmeticConfig(BaseConfig):
    """Configuration for modular arithmetic training experiments."""
    strategy: str = 'all'  # 'stride', 'random', 'fixed-random', 'target', or 'all'
    p: int = 9973
    train_size: int = 300000
    test_size: int = 1000000
    batch_size: int = 256
    lr: float = 1e-3
    min_lr: float = 5e-7
    epochs: int = 5000
    embed_dim: int = 256
    num_heads: int = 4
    layers: int = 2
    weight_decay: float = 0.1
    optimizer: str = 'adamw'  # 'adamw' or 'adam'

    # Override BaseConfig defaults for epoch-based experiment
    eval_every: int = 1              # evaluate every epoch
    snapshot_every: int = 10         # snapshot every 10 epochs
    checkpoint_every: int = 50       # checkpoint every 50 epochs

    stride: int | None = None          # stride for 'stride' ordering (default: floor(sqrt(p)))
    target_acc: float = 99.5

    def __post_init__(self):
        super().__post_init__()
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.p <= 0:
            raise ValueError(f"p must be > 0, got {self.p}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.embed_dim}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.layers <= 0:
            raise ValueError(f"layers must be > 0, got {self.layers}")
        if self.optimizer not in ('adamw', 'adam'):
            raise ValueError(f"optimizer must be 'adamw' or 'adam', got '{self.optimizer}'")
