"""Shared data pool and dataset configuration infrastructure.

DataPool and FixedDataPool manage streaming/buffered access to large datasets.
DATASET_CONFIGS provides standard dataset definitions.

Extracted from experiment_curriculum_test.py and experiment_guided_llm_feature_test2.py
where these were duplicated.
"""

import random

from datasets import load_dataset


# Dataset options for HuggingFace datasets
DATASET_CONFIGS = {
    'wikitext-2': {
        'name': 'wikitext',
        'config': 'wikitext-2-raw-v1',
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': 'Toy dataset - 2M tokens (NOT recommended for real training)',
    },
    'wikitext-103': {
        'name': 'wikitext',
        'config': 'wikitext-103-raw-v1',
        'split': 'train',
        'text_field': 'text',
        'streaming': False,
        'description': 'Small dataset - 103M tokens (marginal for grammar)',
    },
    'wikipedia': {
        'name': 'wikimedia/wikipedia',
        'config': '20231101.en',
        'split': 'train',
        'text_field': 'text',
        'streaming': True,  # Stream to avoid massive download
        'description': 'Full English Wikipedia - ~3B tokens (RECOMMENDED)',
    },
    'openwebtext': {
        'name': 'openwebtext',
        'config': None,
        'split': 'train',
        'text_field': 'text',
        'streaming': True,
        'description': 'OpenWebText - ~8B tokens (good but larger disk usage)',
    },
}


class DataPool:
    """
    Manages a pool of training examples, supporting both full-load and streaming modes.
    For large datasets like Wikipedia, maintains a rolling buffer of examples.

    Args:
        dataset_config: Dict from DATASET_CONFIGS with dataset loading parameters.
        filter_terms: Optional list of terms to filter out from examples.
        buffer_size: Maximum buffer size for streaming mode.
        min_length: Minimum character length for examples.
        console: Optional OLConsole or Console for logging output.
    """
    def __init__(self, dataset_config, filter_terms=None, buffer_size=50000, min_length=100, console=None):
        self.config = dataset_config
        self.filter_terms = filter_terms or []
        self.buffer_size = buffer_size
        self.min_length = min_length
        self.buffer = []
        self.stream_iterator = None
        self.streaming = dataset_config.get('streaming', False)
        self.exhausted = False
        self._console = console

        self._load_dataset()

    def _print(self, msg):
        if self._console is not None:
            self._console.print(msg)

    def _load_dataset(self):
        """Load the dataset (full or streaming mode)."""
        self._print(f"[status]Loading dataset:[/status] [metric.value]{self.config['name']}[/metric.value] ({self.config.get('config', 'default')})")
        self._print(f"[label]Mode:[/label] [metric.value]{'streaming' if self.streaming else 'full load'}[/metric.value]")

        load_kwargs = {
            'path': self.config['name'],
            'split': self.config['split'],
        }
        if self.config.get('config'):
            load_kwargs['name'] = self.config['config']
        if self.streaming:
            load_kwargs['streaming'] = True

        dataset = load_dataset(**load_kwargs)

        if self.streaming:
            self.stream_iterator = iter(dataset)
            self._fill_buffer()
        else:
            # Full load - filter once
            self._print("[status]Filtering dataset...[/status]")
            text_field = self.config['text_field']
            self.buffer = [
                ex[text_field] for ex in dataset
                if len(ex[text_field]) >= self.min_length
                and not any(term in ex[text_field] for term in self.filter_terms)
            ]
            self._print(f"[status]Loaded[/status] [value.count]{len(self.buffer):,}[/value.count] [detail]examples[/detail]")

    def _fill_buffer(self):
        """Fill the streaming buffer with new examples."""
        if self.exhausted:
            return

        old_size = len(self.buffer)
        text_field = self.config['text_field']
        attempts = 0
        max_attempts = self.buffer_size * 10  # Avoid infinite loop

        while len(self.buffer) < self.buffer_size and attempts < max_attempts:
            try:
                example = next(self.stream_iterator)
                text = example[text_field]
                attempts += 1

                # Filter
                if len(text) < self.min_length:
                    continue
                if any(term in text for term in self.filter_terms):
                    continue

                self.buffer.append(text)

            except StopIteration:
                if self._console is not None:
                    self._console.print_warning("Dataset exhausted - restarting stream")
                # Restart the stream
                load_kwargs = {
                    'path': self.config['name'],
                    'split': self.config['split'],
                    'streaming': True,
                }
                if self.config.get('config'):
                    load_kwargs['name'] = self.config['config']
                dataset = load_dataset(**load_kwargs)
                self.stream_iterator = iter(dataset)

        added = len(self.buffer) - old_size
        self._print(f"[label]Buffer:[/label] {old_size:,} → [value.count]{len(self.buffer):,}[/value.count] [detail](+{added:,} examples)[/detail]")

    def sample(self, n):
        """Sample n random examples from the pool. Used for analysis, not training."""
        # Ensure we have enough samples
        if len(self.buffer) < n:
            if self.streaming:
                self._fill_buffer()
            if len(self.buffer) < n:
                n = len(self.buffer)

        # Sample random indices
        indices = random.sample(range(len(self.buffer)), n)
        samples = [(i, self.buffer[i]) for i in indices]

        if self.streaming:
            # Streaming mode: remove used samples to avoid reuse
            for i in sorted(indices, reverse=True):
                self.buffer.pop(i)

            # Refill buffer when it drops to half capacity
            if len(self.buffer) <= self.buffer_size // 2:
                self._fill_buffer()
        # Non-streaming mode: keep all samples, allow reuse (can't refill anyway)

        return samples

    def take(self, n):
        """
        Take the next n examples sequentially from the buffer.
        Refills from stream if necessary. Used by FixedDataPool.

        Args:
            n: Number of examples to take

        Returns:
            List of text examples
        """
        results = []

        while len(results) < n:
            # Refill if buffer is low
            if len(self.buffer) == 0:
                if self.streaming:
                    self._fill_buffer()
                if len(self.buffer) == 0:
                    if self._console is not None:
                        self._console.print_warning(f"DataPool exhausted after {len(results)} examples")
                    break

            # Refill proactively when below threshold
            if self.streaming and len(self.buffer) <= self.buffer_size // 2:
                self._fill_buffer()

            # Take from front of buffer
            results.append(self.buffer.pop(0))

        return results

    def analyze_content(self, terms, sample_size=10000):
        """Analyze how much content matches given terms."""
        from .display import display_content_analysis
        display_content_analysis(self._console, self.buffer, terms, sample_size)

    def __len__(self):
        return len(self.buffer)


class FixedDataPool:
    """
    A pool of examples for training that refills from a DataPool source.
    Maintains examples within min/max thresholds for memory efficiency.
    Tracks cumulative examples taken to ensure all conditions see same total data.
    Sampling is deterministic based on step number for reproducibility.

    Args:
        source_pool: DataPool to draw examples from.
        batch_size: Training batch size (used to compute min/max thresholds).
        total_examples_limit: Maximum examples to ever take from source.
        seed: Base seed for reproducibility.
        console: Optional OLConsole or Console for logging output.
    """
    def __init__(self, source_pool, batch_size, total_examples_limit, seed=42, console=None):
        self.source_pool = source_pool
        self.base_seed = seed
        self.batch_size = batch_size
        self.total_examples_limit = total_examples_limit
        self._console = console

        # Track how many examples we've cumulatively taken from source
        self.cumulative_taken = 0

        # Thresholds based on batch_size (for memory efficiency)
        # max = 1000 steps worth of examples, but never more than total limit
        # min = half of max
        self.max_examples = min(1000 * batch_size, total_examples_limit)
        self.min_examples = self.max_examples // 2

        self.examples = []
        self.remaining_indices = set()

        self._print(f"[status]Creating FixedDataPool[/status] (min=[metric.value]{self.min_examples:,}[/metric.value], max=[metric.value]{self.max_examples:,}[/metric.value], limit=[metric.value]{self.total_examples_limit:,}[/metric.value])")

        # Initial fill to max (but not exceeding limit)
        self._refill()

    def _print(self, msg):
        if self._console is not None:
            self._console.print(msg)

    def _refill(self):
        """Refill from source DataPool up to max_examples, respecting total limit."""
        if self.source_pool is None:
            return

        # How many more can we ever take?
        remaining_budget = self.total_examples_limit - self.cumulative_taken
        if remaining_budget <= 0:
            return  # We've taken all we're allowed to

        # How many do we want to add to reach max?
        wanted = self.max_examples - len(self.remaining_indices)
        if wanted <= 0:
            return

        # Don't exceed our remaining budget
        to_take = min(wanted, remaining_budget)

        old_count = len(self.remaining_indices)

        # Take sequential examples from source (DataPool.take handles its own refilling)
        new_examples = self.source_pool.take(to_take)

        # Add to our pool
        for text in new_examples:
            new_idx = len(self.examples)
            self.examples.append(text)
            self.remaining_indices.add(new_idx)

        self.cumulative_taken += len(new_examples)

        added = len(self.remaining_indices) - old_count
        if added > 0:
            self._print(f"[label]FixedDataPool:[/label] {old_count:,} → [value.count]{len(self.remaining_indices):,}[/value.count] [detail]remaining (+{added:,} new, {self.cumulative_taken:,}/{self.total_examples_limit:,} total taken)[/detail]")

    def sample(self, n, step):
        """
        Sample n examples from remaining pool.
        Uses step-based seeding for reproducibility.
        Returns fewer than n if pool is running low (near end of training).

        Args:
            n: Number of candidates to sample
            step: Current training step (used for deterministic seeding)

        Returns:
            List of (index, text) tuples
        """
        # Refill if below minimum threshold (and we haven't hit our limit)
        if len(self.remaining_indices) < self.min_examples:
            self._refill()

        if len(self.remaining_indices) == 0:
            if self._console is not None:
                self._console.print_warning("FixedDataPool empty")
            return []

        # Deterministic sampling based on step
        rng = random.Random(self.base_seed + step)

        remaining_list = sorted(self.remaining_indices)  # Sort for determinism
        n = min(n, len(remaining_list))  # Return fewer if pool is low
        sampled_indices = rng.sample(remaining_list, n)

        return [(i, self.examples[i]) for i in sampled_indices]

    def remove(self, index):
        """Remove an example from the remaining pool (after selection)."""
        self.remaining_indices.discard(index)

    def __len__(self):
        return len(self.remaining_indices)
