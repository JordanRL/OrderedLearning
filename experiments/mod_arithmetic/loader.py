"""Dataset loaders for modular arithmetic experiments."""

from framework import DatasetLoader
from .dataset import SparseModularDataset, GPUBatchIterator


class ModArithmeticLoader(DatasetLoader):
    """Creates GPU-native batch iterators for modular arithmetic data.

    Handles stride, target, random, and fixed-random orderings.
    Uses GPUBatchIterator to bypass DataLoader CPU overhead — all index
    generation and batch gathering happens directly on GPU.
    """

    def __init__(self, strategy='stride', p=9973, batch_size=256, seed=42, stride=None):
        self.strategy = strategy
        self.p = p
        self.batch_size = batch_size
        self.seed = seed
        self.stride = stride

    def load(self, raw_data, config, **kwargs):
        """Return one or more batch iterators based on strategy.

        For 'alternating': returns a list [stride_iter, target_iter].
        For all others: returns a single-element list.

        The runner handles iteration over multiple loaders.
        """
        if self.strategy == 'alternating':
            ds_stride = SparseModularDataset(raw_data, mode='structured_stride', p=self.p, stride=self.stride)
            ds_target = SparseModularDataset(raw_data, mode='structured_target', p=self.p)
            return [
                GPUBatchIterator(ds_stride, batch_size=self.batch_size),
                GPUBatchIterator(ds_target, batch_size=self.batch_size),
            ]
        elif self.strategy == 'target':
            ds = SparseModularDataset(raw_data, mode='structured_target', p=self.p)
            return [GPUBatchIterator(ds, batch_size=self.batch_size)]
        elif self.strategy == 'stride':
            ds = SparseModularDataset(raw_data, mode='structured_stride', p=self.p, stride=self.stride)
            return [GPUBatchIterator(ds, batch_size=self.batch_size)]
        elif self.strategy == 'fixed-random':
            ds = SparseModularDataset(raw_data, mode='random', p=self.p)
            return [GPUBatchIterator(ds, batch_size=self.batch_size)]
        elif self.strategy == 'resonant':
            ds = SparseModularDataset(raw_data, mode='preordered', p=self.p)
            return [GPUBatchIterator(ds, batch_size=self.batch_size)]
        else:  # random
            ds = SparseModularDataset(raw_data, mode='random', p=self.p)
            return [GPUBatchIterator(ds, batch_size=self.batch_size, shuffle=True, seed=self.seed)]
