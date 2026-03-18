"""Configuration for modular polynomial experiments."""

from dataclasses import dataclass, field

from framework import BaseConfig


@dataclass
class ModPolynomialConfig(BaseConfig):
    """Configuration for x^3 + xy^2 + y (mod p) training experiments."""
    strategy: str = 'all'  # 'stride', 'random', 'fixed-random', 'target', or 'all'
    p: int = 97
    train_fraction: float = 0.5  # fraction of p^2 pairs used for training
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 50000
    embed_dim: int = 128
    num_heads: int = 4
    layers: list[int] = field(default_factory=lambda: [2])
    weight_decay: float = 0.1
    optimizer: str = 'adamw'  # 'adamw' or 'adam'

    # Override BaseConfig defaults for epoch-based experiment
    eval_every: int = 100
    snapshot_every: int = 100
    checkpoint_every: int = 1000

    stride: int | None = None  # stride for 'stride' ordering (default: floor(sqrt(p)))
    target_acc: float = 99.0

    def __post_init__(self):
        super().__post_init__()
        # Normalize single int to list
        if isinstance(self.layers, int):
            self.layers = [self.layers]
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
        for n in self.layers:
            if n <= 0:
                raise ValueError(f"All layer counts must be > 0, got {n}")
        if self.optimizer not in ('adamw', 'adam'):
            raise ValueError(f"optimizer must be 'adamw' or 'adam', got '{self.optimizer}'")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError(f"train_fraction must be in (0, 1), got {self.train_fraction}")
