"""Tests for framework/data/pools.py — DataPool and FixedDataPool."""

import random
from unittest.mock import MagicMock, patch

from framework.data.pools import DataPool, FixedDataPool, DATASET_CONFIGS


class MockDataPool:
    """Minimal mock of DataPool that provides deterministic text examples."""

    def __init__(self, examples=None, n=100):
        self._examples = examples or [f"Example text number {i}" for i in range(n)]
        self._idx = 0

    def take(self, n):
        result = self._examples[self._idx:self._idx + n]
        self._idx += len(result)
        return result


class TestDatasetConfigs:

    def test_has_expected_configs(self):
        assert 'wikitext-2' in DATASET_CONFIGS
        assert 'wikitext-103' in DATASET_CONFIGS
        assert 'wikipedia' in DATASET_CONFIGS

    def test_configs_have_required_keys(self):
        required = {'name', 'split', 'text_field'}
        for name, cfg in DATASET_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"{name} missing keys"


class TestFixedDataPool:

    def test_initial_fill(self):
        """Pool fills on construction to the total limit."""
        source = MockDataPool()
        pool = FixedDataPool(source, batch_size=4, total_examples_limit=100, seed=42)
        assert len(pool) == 100

    def test_sample_returns_tuples(self):
        """sample returns list of (index, text) tuples."""
        source = MockDataPool()
        pool = FixedDataPool(source, batch_size=4, total_examples_limit=100, seed=42)
        samples = pool.sample(5, step=0)
        assert len(samples) == 5
        for idx, text in samples:
            assert isinstance(idx, int)
            assert isinstance(text, str)

    def test_sample_deterministic(self):
        """Same step produces same samples."""
        source1 = MockDataPool()
        pool1 = FixedDataPool(source1, batch_size=4, total_examples_limit=100, seed=42)
        samples1 = pool1.sample(5, step=0)

        source2 = MockDataPool()
        pool2 = FixedDataPool(source2, batch_size=4, total_examples_limit=100, seed=42)
        samples2 = pool2.sample(5, step=0)

        assert [s[0] for s in samples1] == [s[0] for s in samples2]

    def test_different_steps_different_samples(self):
        """Different steps produce different samples."""
        source = MockDataPool()
        pool = FixedDataPool(source, batch_size=4, total_examples_limit=100, seed=42)
        s1 = pool.sample(5, step=0)
        s2 = pool.sample(5, step=1)
        # Indices should differ (with very high probability)
        assert [s[0] for s in s1] != [s[0] for s in s2]

    def test_remove_shrinks_pool(self):
        """remove() removes an index from the pool."""
        source = MockDataPool()
        pool = FixedDataPool(source, batch_size=4, total_examples_limit=100, seed=42)
        initial_len = len(pool)
        samples = pool.sample(1, step=0)
        pool.remove(samples[0][0])
        assert len(pool) == initial_len - 1

    def test_respects_total_limit(self):
        """Cumulative examples taken never exceeds total_examples_limit."""
        source = MockDataPool(n=200)
        pool = FixedDataPool(source, batch_size=4, total_examples_limit=50, seed=42)
        assert pool.cumulative_taken <= 50

    def test_refill_on_low(self):
        """Pool refills when below min_examples threshold."""
        source = MockDataPool(n=500)
        pool = FixedDataPool(source, batch_size=2, total_examples_limit=500, seed=42)
        initial_count = len(pool)
        # Remove many to drop below min
        for idx in list(pool.remaining_indices)[:initial_count - 1]:
            pool.remove(idx)
        # Sample should trigger refill
        pool.sample(1, step=0)

    def test_empty_pool_returns_empty(self):
        """Returns empty list when pool is exhausted."""
        source = MockDataPool(n=5)
        pool = FixedDataPool(source, batch_size=1, total_examples_limit=5, seed=42)
        # Remove all
        for idx in list(pool.remaining_indices):
            pool.remove(idx)
        samples = pool.sample(5, step=0)
        assert samples == []

    def test_sample_fewer_than_requested(self):
        """Returns fewer than n when pool is small."""
        source = MockDataPool(n=3)
        pool = FixedDataPool(source, batch_size=1, total_examples_limit=3, seed=42)
        samples = pool.sample(100, step=0)
        assert len(samples) <= 3

    def test_no_source_pool(self):
        """Handles None source_pool gracefully."""
        pool = FixedDataPool.__new__(FixedDataPool)
        pool.source_pool = None
        pool.base_seed = 42
        pool.batch_size = 1
        pool.total_examples_limit = 10
        pool.max_examples = 10
        pool.min_examples = 5
        pool.examples = ["a", "b", "c"]
        pool.remaining_indices = {0, 1, 2}
        pool.cumulative_taken = 3
        pool._console = None
        # _refill should be a no-op
        pool._refill()
        assert len(pool.remaining_indices) == 3


# ==================================================================
# DataPool — full-load (non-streaming) mode
# ==================================================================

def _make_mock_dataset(texts):
    """Create a mock HuggingFace dataset from a list of text strings."""
    return [{'text': t} for t in texts]


def _make_data_pool(texts=None, streaming=False, min_length=10, **kwargs):
    """Build a DataPool with mocked load_dataset."""
    if texts is None:
        texts = [f"This is example text number {i} with enough length" for i in range(50)]

    config = {
        'name': 'test_dataset',
        'config': None,
        'split': 'train',
        'text_field': 'text',
        'streaming': streaming,
    }

    mock_dataset = _make_mock_dataset(texts)

    with patch('framework.data.pools.load_dataset', return_value=mock_dataset):
        pool = DataPool(config, min_length=min_length, **kwargs)

    return pool


class TestDataPoolFullLoad:

    def test_loads_all_examples(self):
        """Non-streaming mode loads all qualifying examples into buffer."""
        pool = _make_data_pool()
        assert len(pool) == 50

    def test_filters_by_min_length(self):
        """Examples shorter than min_length are excluded."""
        texts = ["short", "This is a long enough example to pass the filter"]
        pool = _make_data_pool(texts=texts, min_length=20)
        assert len(pool) == 1

    def test_filters_by_filter_terms(self):
        """Examples containing filter terms are excluded."""
        texts = [
            "This is a clean example with enough length",
            "This example contains BAD_TERM and is excluded",
            "Another clean example that should be included too",
        ]
        pool = _make_data_pool(texts=texts, min_length=10, filter_terms=['BAD_TERM'])
        assert len(pool) == 2

    def test_sample_returns_index_text_tuples(self):
        """sample() returns list of (index, text) tuples."""
        pool = _make_data_pool()
        samples = pool.sample(5)
        assert len(samples) == 5
        for idx, text in samples:
            assert isinstance(idx, int)
            assert isinstance(text, str)

    def test_sample_non_streaming_allows_reuse(self):
        """Non-streaming sample doesn't remove items from buffer."""
        pool = _make_data_pool()
        initial_len = len(pool)
        pool.sample(5)
        assert len(pool) == initial_len

    def test_sample_fewer_when_not_enough(self):
        """sample() returns fewer than requested when buffer is small."""
        texts = ["Long enough example text number one to pass"]
        pool = _make_data_pool(texts=texts, min_length=10)
        samples = pool.sample(100)
        assert len(samples) <= 1

    def test_take_returns_sequential(self):
        """take() returns examples in order from front of buffer."""
        texts = [f"Example text number {i} with enough chars" for i in range(10)]
        pool = _make_data_pool(texts=texts, min_length=10)
        taken = pool.take(3)
        assert len(taken) == 3
        # Buffer shrinks by 3 (take pops from front)
        assert len(pool) == 7

    def test_take_more_than_available(self):
        """take() returns whatever is available when buffer < n."""
        texts = ["Long enough example text that passes the min length filter"]
        pool = _make_data_pool(texts=texts, min_length=10)
        taken = pool.take(100)
        assert len(taken) == 1

    def test_print_with_console(self):
        """_print outputs to console when one is provided."""
        console = MagicMock()
        pool = _make_data_pool(console=console)
        assert console.print.called


class TestDataPoolStreaming:

    def _make_streaming_pool(self, texts=None, min_length=10, buffer_size=20):
        """Build a streaming DataPool with mock iterator.

        Note: load_dataset is called twice in streaming mode — once in __init__
        and again when the stream is restarted in _fill_buffer. We provide
        enough texts to fill the buffer on the first call so the second
        call (stream restart) is never reached.
        """
        if texts is None:
            texts = [f"Streaming example number {i} long enough text" for i in range(50)]

        config = {
            'name': 'test_stream',
            'config': 'en',
            'split': 'train',
            'text_field': 'text',
            'streaming': True,
        }

        mock_records = _make_mock_dataset(texts)

        with patch('framework.data.pools.load_dataset', return_value=mock_records):
            pool = DataPool(config, min_length=min_length, buffer_size=buffer_size)

        return pool

    def test_streaming_fills_buffer(self):
        """Streaming mode fills buffer up to buffer_size."""
        pool = self._make_streaming_pool(buffer_size=20)
        assert len(pool) == 20

    def test_streaming_sample_removes_items(self):
        """Streaming sample removes used items from buffer."""
        pool = self._make_streaming_pool(buffer_size=20)
        initial_len = len(pool)
        pool.sample(5)
        # Items were removed (streaming mode)
        assert len(pool) < initial_len

    def test_streaming_take_pops_front(self):
        """Streaming take removes from front of buffer."""
        pool = self._make_streaming_pool(buffer_size=20)
        taken = pool.take(5)
        assert len(taken) == 5
        assert len(pool) == 15

    def test_streaming_filters_by_min_length(self):
        """Short examples in stream are skipped."""
        # Provide enough long texts to fill buffer without triggering restart
        texts = ["hi"] * 10 + [f"Long enough streaming text number {i}" for i in range(30)]
        pool = self._make_streaming_pool(texts=texts, min_length=20, buffer_size=5)
        assert len(pool) == 5
