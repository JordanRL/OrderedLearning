"""Reproducibility utilities: seeds, determinism, and environment tracking."""

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
