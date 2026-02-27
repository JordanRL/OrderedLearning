"""Model configurations and scheduler utilities.

Provides GPT-2 model configuration presets, learning rate scheduler,
and predictive coding model hierarchy.
"""

from .models import MODEL_CONFIGS, get_lr_scheduler
from .predictive_coding import PCLayer, PCLayerConfig, PredictiveCodingNetwork

__all__ = [
    'MODEL_CONFIGS', 'get_lr_scheduler',
    'PCLayer', 'PCLayerConfig', 'PredictiveCodingNetwork',
]
