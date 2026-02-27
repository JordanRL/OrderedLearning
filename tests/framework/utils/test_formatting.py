"""Tests for framework/utils/formatting.py — human-readable formatting, JSON default, pad_sequences."""

import torch
import pytest
from framework.utils.formatting import format_human_readable, format_bytes, _json_default, pad_sequences


class TestFormatHumanReadable:

    def test_small_number(self):
        """Numbers < 1000 pass through as string."""
        assert format_human_readable(42) == "42"

    def test_thousands(self):
        """Numbers >= 1000 use K suffix."""
        assert format_human_readable(1500) == "1.50K"

    def test_millions(self):
        """Numbers >= 1M use M suffix."""
        assert format_human_readable(2_500_000) == "2.50M"

    def test_billions(self):
        """Numbers >= 1B use B suffix."""
        assert format_human_readable(3_000_000_000) == "3.00B"


class TestFormatBytes:

    def test_bytes_range(self):
        """Small counts show 'bytes' suffix."""
        assert format_bytes(512) == "512 bytes"

    def test_kilobytes(self):
        """Values >= 1024 show KB."""
        result = format_bytes(2048)
        assert "KB" in result


class TestJsonDefault:

    def test_torch_scalar(self):
        """Torch scalar tensors are converted via .item()."""
        t = torch.tensor(3.14)
        result = _json_default(t)
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-5

    def test_numpy_scalar(self):
        """Numpy scalars are converted via .item()."""
        import numpy as np
        val = np.float64(2.5)
        result = _json_default(val)
        assert isinstance(result, float)
        assert result == 2.5

    def test_fallback_to_str(self):
        """Unrecognized types fall through to str()."""
        result = _json_default(object())
        assert isinstance(result, str)


class TestPadSequences:

    def test_equal_length_no_padding(self):
        """Equal-length sequences produce no padding."""
        ids = [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]
        masks = [torch.ones(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)]
        batch_ids, batch_mask, batch_labels = pad_sequences(ids, masks, pad_token_id=0, device='cpu')
        assert batch_ids.shape == (2, 3)
        assert (batch_mask == 1).all()

    def test_unequal_lengths_pads_shorter(self):
        """Shorter sequences are padded; padding positions get -100 in labels."""
        ids = [torch.tensor([[1, 2]]), torch.tensor([[4, 5, 6]])]
        masks = [torch.ones(1, 2, dtype=torch.long), torch.ones(1, 3, dtype=torch.long)]
        batch_ids, batch_mask, batch_labels = pad_sequences(ids, masks, pad_token_id=0, device='cpu')
        assert batch_ids.shape == (2, 3)
        assert batch_ids[0, 2] == 0          # padding token
        assert batch_labels[0, 2] == -100    # label mask
        assert batch_mask[0, 2] == 0         # attention mask off
