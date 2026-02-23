"""Shared utility functions for experiment framework.

Extracted from experiment files where these were duplicated.
"""

import os
import platform
import random

import numpy as np
import torch


def set_seeds(seed: int):
    """Set random seeds for reproducibility across all backends."""
    # PYTHONHASHSEED only affects hash randomization when set before the
    # interpreter starts, but we set it here for documentation/signaling
    # and for libraries that check it at runtime.
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_determinism(enabled: bool = True):
    """Configure torch deterministic algorithm enforcement.

    When enabled: forces deterministic CUDA kernels, disables cudnn
    benchmark, and disables non-deterministic SDPA backends.
    When disabled: allows non-deterministic kernels and enables
    cudnn benchmark for performance.
    """
    if enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def get_environment_info() -> dict:
    """Collect environment metadata relevant to reproducibility.

    Returns a dict suitable for JSON serialization and embedding in
    saved model/config files.
    """
    info = {
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = str(torch.backends.cudnn.version())
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_capability'] = '.'.join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
        info['gpu_count'] = torch.cuda.device_count()

    # Determinism settings
    info['float32_matmul_precision'] = torch.get_float32_matmul_precision()
    info['cudnn_deterministic'] = torch.backends.cudnn.deterministic
    info['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    info['cublas_workspace_config'] = os.environ.get(
        'CUBLAS_WORKSPACE_CONFIG', ''
    )

    return info


# Keys in environment info that affect numerical reproducibility.
# Differences in these between a saved model and the current environment
# mean the weights may produce different training dynamics.
_REPRODUCIBILITY_KEYS = {
    'torch_version': 'PyTorch version',
    'cuda_version': 'CUDA version',
    'cudnn_version': 'cuDNN version',
    'gpu_name': 'GPU model',
    'gpu_capability': 'GPU compute capability',
    'float32_matmul_precision': 'float32 matmul precision',
}


def check_environment_compatibility(
    saved_env: dict,
    current_env: dict | None = None,
) -> list[str]:
    """Compare saved environment info against current environment.

    Returns a list of human-readable warning strings for each
    reproducibility-relevant difference. Empty list means compatible.
    """
    if current_env is None:
        current_env = get_environment_info()

    warnings = []
    for key, label in _REPRODUCIBILITY_KEYS.items():
        saved_val = saved_env.get(key)
        current_val = current_env.get(key)
        if saved_val is None or current_val is None:
            continue
        if str(saved_val) != str(current_val):
            warnings.append(
                f"{label}: saved={saved_val}, current={current_val}"
            )
    return warnings


def snapshot_params(model):
    """Clone current parameter values to CPU for trajectory recording."""
    return {
        name: param.data.clone().cpu()
        for name, param in model.named_parameters()
    }


def get_gradient_vector(model):
    """Extract flattened gradient from model parameters."""
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    return torch.cat(grads) if grads else None


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return torch.dot(a, b) / (a.norm() * b.norm() + 1e-8)


def pad_sequences(input_ids_list, attention_masks_list, pad_token_id, device):
    """
    Pad a list of sequences to the same length for batched processing.

    Args:
        input_ids_list: List of [1, seq_len] tensors
        attention_masks_list: List of [1, seq_len] tensors
        pad_token_id: Token ID to use for padding
        device: Device to place tensors on

    Returns:
        batch_input_ids: [N, max_len] tensor
        batch_attention_mask: [N, max_len] tensor
        batch_labels: [N, max_len] tensor (with -100 for padding positions)
    """
    max_len = max(ids.size(1) for ids in input_ids_list)

    padded_inputs = []
    padded_masks = []

    for input_ids, attn_mask in zip(input_ids_list, attention_masks_list):
        seq_len = input_ids.size(1)
        if seq_len < max_len:
            pad_len = max_len - seq_len
            input_padding = torch.full((1, pad_len), pad_token_id, dtype=input_ids.dtype, device=device)
            mask_padding = torch.zeros((1, pad_len), dtype=attn_mask.dtype, device=device)
            input_ids = torch.cat([input_ids, input_padding], dim=1)
            attn_mask = torch.cat([attn_mask, mask_padding], dim=1)
        padded_inputs.append(input_ids)
        padded_masks.append(attn_mask)

    batch_input_ids = torch.cat(padded_inputs, dim=0)
    batch_attention_mask = torch.cat(padded_masks, dim=0)

    # Labels: copy input_ids but mask padding positions
    batch_labels = batch_input_ids.clone()
    batch_labels[batch_attention_mask == 0] = -100

    return batch_input_ids, batch_attention_mask, batch_labels


def format_human_readable(num: int) -> str:
    """Format a number with human-readable suffix (K, M, B)."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def format_bytes(num_bytes: int) -> str:
    """Format bytes with human-readable suffix (KB, MB, GB)."""
    if num_bytes >= 1_073_741_824:  # 1 GB
        return f"{num_bytes / 1_073_741_824:.2f} GB"
    elif num_bytes >= 1_048_576:  # 1 MB
        return f"{num_bytes / 1_048_576:.2f} MB"
    elif num_bytes >= 1_024:  # 1 KB
        return f"{num_bytes / 1_024:.2f} KB"
    else:
        return f"{num_bytes} bytes"
