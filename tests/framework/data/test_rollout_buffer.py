"""Tests for framework/data/rollout_buffer.py — RL rollout storage and batching."""

import pytest
import torch

from framework.data.rollout_buffer import RolloutBuffer, RolloutBatch


def _fill_buffer(buffer, n=8, obs_dim=4):
    """Add n transitions to a buffer."""
    for i in range(n):
        buffer.add(
            obs=torch.randn(obs_dim),
            action=torch.tensor(i % 2),
            reward=float(i),
            done=float(i == n - 1),  # terminal on last step
            log_prob=torch.tensor(-0.5),
            value=float(i) * 0.1,
        )


class TestRolloutBufferStorage:

    def test_add_tracks_length(self):
        """add() increments len() correctly."""
        buffer = RolloutBuffer(buffer_size=16)
        assert len(buffer) == 0
        _fill_buffer(buffer, n=5)
        assert len(buffer) == 5

    def test_full_property(self):
        """full is True when len >= buffer_size."""
        buffer = RolloutBuffer(buffer_size=4)
        _fill_buffer(buffer, n=3)
        assert not buffer.full
        _fill_buffer(buffer, n=1)
        assert buffer.full

    def test_reset_clears_all(self):
        """reset() clears all storage and computed returns."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=5)
        buffer.compute_returns(last_value=0.0)
        assert buffer.advantages is not None
        buffer.reset()
        assert len(buffer) == 0
        assert buffer.advantages is None
        assert buffer.returns is None


class TestRolloutBufferReturns:

    def test_compute_returns_produces_tensors(self):
        """compute_returns() sets advantages and returns as tensors."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=6)
        buffer.compute_returns(last_value=0.5)
        assert isinstance(buffer.advantages, torch.Tensor)
        assert isinstance(buffer.returns, torch.Tensor)
        assert buffer.advantages.shape == (6,)
        assert buffer.returns.shape == (6,)

    def test_returns_equals_advantages_plus_values(self):
        """returns = advantages + values (by construction in GAE)."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=4)
        buffer.compute_returns(last_value=0.0)
        values = torch.tensor(buffer.values, dtype=torch.float32)
        expected = buffer.advantages + values
        assert torch.allclose(buffer.returns, expected)


class TestRolloutBufferBatching:

    def test_get_batches_yields_rollout_batches(self):
        """get_batches() yields RolloutBatch instances with correct shapes."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=8, obs_dim=4)
        buffer.compute_returns(last_value=0.0)
        batches = list(buffer.get_batches(batch_size=4))
        assert len(batches) == 2
        for batch in batches:
            assert isinstance(batch, RolloutBatch)
            assert batch.observations.shape == (4, 4)
            assert batch.actions.shape == (4,)
            assert batch.old_log_probs.shape == (4,)
            assert batch.advantages.shape == (4,)
            assert batch.returns.shape == (4,)

    def test_get_batches_before_compute_returns_raises(self):
        """get_batches() before compute_returns() raises RuntimeError."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=4)
        with pytest.raises(RuntimeError, match="compute_returns"):
            list(buffer.get_batches(batch_size=2))

    def test_get_batches_handles_remainder(self):
        """get_batches() yields a smaller final batch when n is not divisible."""
        buffer = RolloutBuffer(buffer_size=16)
        _fill_buffer(buffer, n=5, obs_dim=4)
        buffer.compute_returns(last_value=0.0)
        batches = list(buffer.get_batches(batch_size=3))
        assert len(batches) == 2
        sizes = [b.observations.shape[0] for b in batches]
        assert sorted(sizes) == [2, 3]
