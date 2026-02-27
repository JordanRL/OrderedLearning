"""Shared utility functions for experiment framework.

Organized into submodules:
- reproducibility: seeds, determinism, environment tracking
- gradients: flattening, similarity, accumulation, snapshots
- formatting: human-readable output, JSON serialization, sequence padding
"""

from .reproducibility import (
    set_seeds, set_determinism,
    get_environment_info, check_environment_compatibility,
)
from .gradients import (
    snapshot_params, get_gradient_vector, cosine_similarity,
    flatten_grads, flatten_params,
    create_accumulator, accumulate, finalize,
)
from .formatting import (
    format_human_readable, format_bytes,
    _json_default, pad_sequences,
)

__all__ = [
    # Reproducibility
    'set_seeds', 'set_determinism',
    'get_environment_info', 'check_environment_compatibility',
    # Gradients
    'snapshot_params', 'get_gradient_vector', 'cosine_similarity',
    'flatten_grads', 'flatten_params',
    'create_accumulator', 'accumulate', 'finalize',
    # Formatting
    'format_human_readable', 'format_bytes',
    '_json_default', 'pad_sequences',
]
