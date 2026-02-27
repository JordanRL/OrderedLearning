"""Checkpoint utilities: save, validate, resume, emergency snapshots, and reference weights.

Provides checkpoint save/validate, EmergencyCheckpoint for signal handling,
resume support for training continuation, and ReferenceWeights for hook
comparison against known solutions.
"""

from .checkpoints import (
    save_checkpoint, validate_checkpoint, EmergencyCheckpoint,
)
from .resume import (
    ResumeInfo,
    check_resume_conflicts, find_latest_checkpoint,
    detect_resume_state, load_config_from_output, load_checkpoint,
)
from .reference_weights import ReferenceWeights, DEFAULT_REFERENCE_PATH

__all__ = [
    'save_checkpoint', 'validate_checkpoint', 'EmergencyCheckpoint',
    'ResumeInfo',
    'check_resume_conflicts', 'find_latest_checkpoint',
    'detect_resume_state', 'load_config_from_output', 'load_checkpoint',
    'ReferenceWeights', 'DEFAULT_REFERENCE_PATH',
]
