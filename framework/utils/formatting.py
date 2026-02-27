"""Formatting and serialization utilities."""

import torch


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


def _json_default(obj):
    """JSON serializer fallback for torch/numpy scalars and arrays."""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


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
