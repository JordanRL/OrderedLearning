"""Dataset classes for modular arithmetic experiments."""

import math

import torch
from torch.utils.data import Dataset


class SparseModularDataset(Dataset):
    """Dataset of (a, b, c) tuples where c = (a + b) mod p.

    Supports different orderings:
    - 'structured_stride': groups by geometric proximity (a % stride)
    - 'structured_target': groups by algebraic target (sum mod p)
    - 'random': no reordering
    """

    def __init__(self, raw_data, mode='random', p=9973, device=None, stride=None):
        if mode == 'structured_stride':
            stride = stride if stride is not None else int(math.sqrt(p))
            final_data = sorted(raw_data, key=lambda x: (x[0] % stride, x[0]))
        elif mode == 'structured_target':
            final_data = sorted(raw_data, key=lambda x: x[2])
        else:
            final_data = raw_data

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data = torch.tensor(final_data, dtype=torch.long, device=device)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


class GPUBatchIterator:
    """GPU-native batch iterator that bypasses DataLoader CPU overhead.

    PyTorch's DataLoader calls __getitem__ individually for each sample
    (CPU-mediated), then collates via torch.stack. This class generates
    indices and gathers batches directly on GPU with a single tensor
    operation per batch.
    """

    def __init__(self, dataset, batch_size, shuffle=False, seed=None):
        self.data = dataset.data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._epoch_seed = seed

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def seed_epoch(self, seed):
        """Set the seed for the next iteration's shuffle."""
        self._epoch_seed = seed

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            g = torch.Generator(device=self.data.device)
            if self._epoch_seed is not None:
                g.manual_seed(self._epoch_seed)
            indices = torch.randperm(n, device=self.data.device, generator=g)
        else:
            indices = torch.arange(n, device=self.data.device)

        for start in range(0, n, self.batch_size):
            yield self.data[indices[start:start + self.batch_size]]
