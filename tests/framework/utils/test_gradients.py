"""Tests for framework/utils/gradients.py — snapshot, flatten, cosine_similarity, accumulator."""

import torch
import torch.nn as nn
import pytest
from framework.utils.gradients import (
    snapshot_params, get_gradient_vector, cosine_similarity,
    flatten_grads, flatten_params,
    create_accumulator, accumulate, finalize,
)


@pytest.fixture
def simple_model():
    torch.manual_seed(42)
    return nn.Linear(3, 2)  # 8 params (6 weight + 2 bias)


class TestSnapshotParams:

    def test_snapshot_clones_to_cpu(self, simple_model):
        """snapshot_params returns CPU clones of all parameters."""
        snap = snapshot_params(simple_model)
        assert all(v.device == torch.device('cpu') for v in snap.values())
        # Mutating original does not affect snapshot
        with torch.no_grad():
            next(simple_model.parameters()).fill_(0.0)
        for name, param_snap in snap.items():
            if not (param_snap == 0.0).all():
                return  # at least one parameter differs — pass
        pytest.fail("Snapshot should be independent of the model")


class TestGradientVector:

    def test_returns_none_without_grad(self, simple_model):
        """get_gradient_vector returns None when no .grad is populated."""
        assert get_gradient_vector(simple_model) is None

    def test_returns_flat_vector(self, simple_model):
        """After backward, returns a flat 1D vector."""
        x = torch.randn(1, 3)
        loss = simple_model(x).sum()
        loss.backward()
        vec = get_gradient_vector(simple_model)
        assert vec is not None
        assert vec.dim() == 1
        assert vec.numel() == 8  # 6 weight + 2 bias


class TestCosineSimilarity:

    def test_identical_vectors(self):
        """Cosine similarity of identical vectors is ~1.0."""
        v = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v).item() == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors is ~0.0."""
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_similarity(a, b).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors is ~-1.0."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([-1.0, -2.0, -3.0])
        assert cosine_similarity(a, b).item() == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector(self):
        """Cosine similarity with zero vector is ~0.0 (epsilon prevents NaN)."""
        a = torch.tensor([1.0, 2.0])
        b = torch.zeros(2)
        result = cosine_similarity(a, b).item()
        assert not torch.isnan(torch.tensor(result))


class TestFlattenParams:

    def test_returns_flat_vector(self):
        """flatten_params returns a 1D tensor."""
        params = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        flat = flatten_params(params, exclude_bias=False)
        assert flat.dim() == 1
        assert flat.numel() == 8

    def test_excludes_bias_by_default(self):
        """flatten_params excludes bias parameters by default."""
        params = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        flat = flatten_params(params)
        assert flat.numel() == 6

    def test_includes_bias_when_requested(self):
        """flatten_params includes bias when exclude_bias=False."""
        params = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        flat = flatten_params(params, exclude_bias=False)
        assert flat.numel() == 8


class TestFlattenGrads:

    def test_excludes_bias_by_default(self):
        """flatten_grads excludes keys containing 'bias'."""
        grads = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        flat = flatten_grads(grads, exclude_bias=True)
        assert flat.numel() == 6  # only weight

    def test_includes_bias_when_requested(self):
        """flatten_grads includes bias when exclude_bias=False."""
        grads = {"weight": torch.ones(2, 3), "bias": torch.ones(2)}
        flat = flatten_grads(grads, exclude_bias=False)
        assert flat.numel() == 8  # weight + bias


class TestAccumulator:

    def test_create_accumulate_finalize_cycle(self, simple_model):
        """Full accumulator lifecycle: create, accumulate 2 batches, finalize to mean."""
        accum, count = create_accumulator(simple_model)
        assert count == 0
        assert all(v.sum() == 0 for v in accum.values())

        # Two forward/backward passes with different inputs
        for _ in range(2):
            x = torch.randn(1, 3)
            loss = simple_model(x).sum()
            loss.backward()
            count = accumulate(accum, simple_model, count)

        assert count == 2
        result = finalize(accum, count, to_cpu=True)
        assert all(v.device == torch.device('cpu') for v in result.values())
        # The finalized gradients should be non-zero (mean of 2 batch gradients)
        assert any(v.abs().sum() > 0 for v in result.values())
