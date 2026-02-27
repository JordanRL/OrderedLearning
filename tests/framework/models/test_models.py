"""Tests for framework/models/models.py — model configs and LR scheduler."""

import torch
from torch.optim import SGD

from framework.models.models import MODEL_CONFIGS, get_lr_scheduler


class TestModelConfigs:

    def test_has_expected_configs(self):
        """MODEL_CONFIGS has tiny, small, and medium."""
        assert 'tiny' in MODEL_CONFIGS
        assert 'small' in MODEL_CONFIGS
        assert 'medium' in MODEL_CONFIGS

    def test_configs_have_required_keys(self):
        """Each config has the keys needed for GPT-2."""
        required = {'vocab_size', 'n_embd', 'n_layer', 'n_head', 'n_positions'}
        for name, config in MODEL_CONFIGS.items():
            assert required.issubset(config.keys()), f"{name} missing keys"


class TestGetLRScheduler:

    def test_returns_scheduler(self):
        """get_lr_scheduler returns a torch scheduler."""
        model = torch.nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=0.01)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, total_steps=100)
        assert hasattr(scheduler, 'step')
        assert hasattr(scheduler, 'get_last_lr')

    def test_warmup_phase(self):
        """During warmup, LR ramps up linearly."""
        model = torch.nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=10, total_steps=100)
        # Step 0: lr_lambda(0) = 0/10 = 0
        lrs = []
        for _ in range(5):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        # LR should be increasing during warmup
        assert lrs[-1] > lrs[0]

    def test_cosine_decay_after_warmup(self):
        """After warmup, LR decays via cosine schedule."""
        model = torch.nn.Linear(4, 2)
        optimizer = SGD(model.parameters(), lr=1.0)
        scheduler = get_lr_scheduler(optimizer, warmup_steps=2, total_steps=20)
        # Step past warmup
        for _ in range(15):
            scheduler.step()
        lr_mid = optimizer.param_groups[0]['lr']
        for _ in range(5):
            scheduler.step()
        lr_end = optimizer.param_groups[0]['lr']
        # LR should be lower at end of training
        assert lr_end <= lr_mid
