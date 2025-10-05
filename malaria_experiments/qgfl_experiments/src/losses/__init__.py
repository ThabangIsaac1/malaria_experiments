"""
QGFL Loss Module
Modular loss functions for malaria detection with class imbalance
"""

from .qgfl_core import QGFLCore
from .qgfl_yolo import QGFLYOLOLoss
from .qgfl_rtdetr import QGFLRTDETRLoss

__all__ = ['QGFLCore', 'QGFLYOLOLoss', 'QGFLRTDETRLoss']
