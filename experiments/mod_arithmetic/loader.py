"""Dataset loaders for modular arithmetic experiments."""

import torch
from torch.utils.data import DataLoader

from framework import DatasetLoader
from .dataset import SparseModularDataset


class ModArithmeticLoader(DatasetLoader):
    """Creates ordered DataLoaders for modular arithmetic data.

    Handles stride, target, random, and fixed-random orderings.
    """

    def __init__(self, strategy='stride', p=9973, batch_size=256, seed=42, stride=None):
        self.strategy = strategy
        self.p = p
        self.batch_size = batch_size
        self.seed = seed
        self.stride = stride

    def load(self, raw_data, config, **kwargs):
        """Return one or more DataLoaders based on strategy.

        For 'alternating': returns a list [stride_loader, target_loader].
        For all others: returns a single DataLoader.

        The runner handles iteration over multiple loaders.
        """
        if self.strategy == 'alternating':
            ds_stride = SparseModularDataset(raw_data, mode='structured_stride', p=self.p, stride=self.stride)
            ds_target = SparseModularDataset(raw_data, mode='structured_target', p=self.p)
            return [
                DataLoader(ds_stride, batch_size=self.batch_size, shuffle=False),
                DataLoader(ds_target, batch_size=self.batch_size, shuffle=False),
            ]
        elif self.strategy == 'target':
            ds = SparseModularDataset(raw_data, mode='structured_target', p=self.p)
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=False)]
        elif self.strategy == 'stride':
            ds = SparseModularDataset(raw_data, mode='structured_stride', p=self.p, stride=self.stride)
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=False)]
        elif self.strategy == 'fixed-random':
            ds = SparseModularDataset(raw_data, mode='random', p=self.p)
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=False)]
        else:  # random
            ds = SparseModularDataset(raw_data, mode='random', p=self.p)
            g = torch.Generator()
            g.manual_seed(self.seed)
            return [DataLoader(ds, batch_size=self.batch_size, shuffle=True, generator=g)]
