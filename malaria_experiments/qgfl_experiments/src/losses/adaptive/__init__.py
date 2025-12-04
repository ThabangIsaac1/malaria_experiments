"""
Dynamic Adaptive QGFL Module

Extends QGFL with epoch-level parameter adaptation based on per-class validation metrics.
Automatically adjusts alpha, gamma, and threshold during training.

Author: Thabang Isaka
Date: 2025-11-21
"""

# Core components (always available)
from .config import AdaptiveQGFLConfig
from .core import DynamicAdaptiveQGFLCore

# Lazy imports for adapters and callbacks (may not exist yet during development)
def __getattr__(name):
    """Lazy import for optional modules"""
    if name == 'DynamicAdaptiveYOLOLoss':
        from .yolo_adapter import DynamicAdaptiveYOLOLoss
        return DynamicAdaptiveYOLOLoss
    elif name == 'DynamicAdaptiveRTDETRLoss':
        from .rtdetr_adapter import DynamicAdaptiveRTDETRLoss
        return DynamicAdaptiveRTDETRLoss
    elif name == 'AdaptiveTrainingCallback':
        from .callbacks import AdaptiveTrainingCallback
        return AdaptiveTrainingCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'AdaptiveQGFLConfig',
    'DynamicAdaptiveQGFLCore',
    'DynamicAdaptiveYOLOLoss',
    'DynamicAdaptiveRTDETRLoss',
    'AdaptiveTrainingCallback',
]
