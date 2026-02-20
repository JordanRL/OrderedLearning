"""Shared model configurations and scheduler utilities.

Extracted from experiment files where these were duplicated across
curriculum, presorted, and guided experiments.
"""

import numpy as np
import torch


# GPT-2 model configurations (used by step-based experiments)
# Note: attn_implementation="eager" allows torch.compile to optimize better
MODEL_CONFIGS = {
    'tiny': dict(
        vocab_size=50257,
        n_embd=384,
        n_layer=6,
        n_head=6,
        n_positions=512,
        loss_type="ForCausalLMLoss",
        attn_implementation="eager",
        # ~49M params, ~2GB VRAM
    ),
    'small': dict(
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_positions=512,
        loss_type="ForCausalLMLoss",
        attn_implementation="eager",
        # ~124M params, ~6GB VRAM (GPT-2 Small equivalent)
    ),
    'medium': dict(
        vocab_size=50257,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        n_positions=512,
        loss_type="ForCausalLMLoss",
        attn_implementation="eager",
        # ~345M params, ~18GB VRAM (GPT-2 Medium equivalent)
    ),
}


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # Handle edge case where warmup_steps >= total_steps
        remaining = total_steps - warmup_steps
        if remaining <= 0:
            return 1.0  # No decay, just stay at full LR
        decay_progress = (step - warmup_steps) / remaining
        return 0.5 * (1 + np.cos(np.pi * decay_progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
