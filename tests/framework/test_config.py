"""Tests for framework/config.py — BaseConfig dataclass and __post_init__ validation."""

import pytest
from framework.config import BaseConfig


class TestBaseConfigDefaults:

    def test_default_construction(self):
        """BaseConfig() creates with all defaults without error."""
        config = BaseConfig()
        assert config.seed == 42
        assert config.eval_every == 500
        assert config.snapshot_every == 1000
        assert config.checkpoint_every == 5000

    def test_custom_values(self):
        """Custom values override defaults."""
        config = BaseConfig(seed=0, eval_every=100)
        assert config.seed == 0
        assert config.eval_every == 100


class TestBaseConfigValidation:

    def test_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed must be >= 0"):
            BaseConfig(seed=-1)

    def test_zero_eval_every_raises(self):
        """eval_every=0 raises ValueError."""
        with pytest.raises(ValueError, match="eval_every must be > 0"):
            BaseConfig(eval_every=0)

    def test_zero_snapshot_every_raises(self):
        """snapshot_every=0 raises ValueError."""
        with pytest.raises(ValueError, match="snapshot_every must be > 0"):
            BaseConfig(snapshot_every=0)

    def test_zero_checkpoint_every_raises(self):
        """checkpoint_every=0 raises ValueError."""
        with pytest.raises(ValueError, match="checkpoint_every must be > 0"):
            BaseConfig(checkpoint_every=0)
