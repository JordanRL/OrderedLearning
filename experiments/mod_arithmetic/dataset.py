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
