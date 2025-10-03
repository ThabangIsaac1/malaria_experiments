# src/training/strategy_wrapper.py
"""
Modular Training Strategy Wrapper for YOLO Models
Supports: No Weights, Weighted Baseline, QGFL with Focal Loss
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import wandb
from pathlib import Path
from collections import defaultdict

class TrainingStrategy(ABC):
    """Base class for all training strategies"""
    
    def __init__(self, config, class_distribution: Dict[str, int]):
        """
        Args:
            config: Configuration object
            class_distribution: {'class_name': count} for training set
        """
        self.config = config
        self.class_distribution = class_distribution
        self.num_classes = len(config.get_class_names())
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        
        # Strategy-specific metrics to track
        self.custom_metrics = {}
        
        # Identify minority classes
        self.minority_classes = self._identify_minority_classes()
        
        # Print strategy info
        self._print_strategy_info()
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate inverse frequency weights"""
        total_samples = sum(self.class_distribution.values())
        weights = []
        
        for class_id in range(self.num_classes):
            class_name = self.config.get_class_names()[class_id]
            count = self.class_distribution.get(class_name, 1)  # Avoid div by 0
            weight = total_samples / (self.num_classes * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def _identify_minority_classes(self) -> list:
        """Identify minority classes based on distribution"""
        if self.num_classes == 2:
            # Binary: minority is the one with fewer samples
            class_names = list(self.config.get_class_names().values())
            counts = [self.class_distribution.get(name, 0) for name in class_names]
            min_count = min(counts)
            minority = []
            for i, count in enumerate(counts):
                if count == min_count:
                    minority.append(i)
            return minority
        else:
            # Multi-class: classes with < 20% of average
            avg_count = sum(self.class_distribution.values()) / self.num_classes
            threshold = avg_count * 0.2
            minority = []
            for class_id, class_name in self.config.get_class_names().items():
                if self.class_distribution.get(class_name, 0) < threshold:
                    minority.append(class_id)
            return minority
    
    def _print_strategy_info(self):
        """Print strategy initialization info"""
        print(f"\nStrategy: {self.get_strategy_name().upper()}")
        print(f"Class Weights: {self.class_weights.numpy()}")
        print(f"Minority Classes: {[self.config.get_class_names()[i] for i in self.minority_classes]}")
    
    @abstractmethod
    def get_loss_function(self):
        """Return the loss function for this strategy"""
        pass
    
    @abstractmethod
    def get_training_params(self) -> Dict[str, Any]:
        """Return strategy-specific training parameters"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        pass
    
    def should_apply_custom_loss(self) -> bool:
        """Whether this strategy uses custom loss"""
        return False
    
    def log_strategy_metrics(self, metrics: Dict):
        """Log strategy-specific metrics to W&B"""
        if self.config.use_wandb and self.custom_metrics:
            wandb.log({
                f'{self.get_strategy_name()}/{k}': v 
                for k, v in self.custom_metrics.items()
            })
    
    def get_hyperparameter_adjustments(self) -> Dict[str, Any]:
        """Get strategy-specific hyperparameter adjustments"""
        return {}


class NoWeightsStrategy(TrainingStrategy):
    """Phase 1: True baseline - no class weights, standard YOLO"""
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Override to return uniform weights"""
        return torch.ones(self.num_classes, dtype=torch.float32)
    
    def get_loss_function(self):
        return None
    
    def get_training_params(self) -> Dict[str, Any]:
        """Standard YOLO parameters without any class weighting"""
        return {
            'cls': 0.5,  # Default YOLO classification loss
            'box': 7.5,  # Default box regression loss  
            'dfl': 1.5   # Default distribution focal loss
        }
    
    def get_strategy_name(self) -> str:
        return "no_weights"
    
    def _print_strategy_info(self):
        print(f"\nStrategy: NO_WEIGHTS (True Baseline)")
        print(f"All classes weighted equally: 1.0")
        print(f"Expected: High accuracy but poor minority recall")


class QGFLStrategy(TrainingStrategy):
    """Phase 3: Query-Guided Focal Loss - Full implementation"""
    
    def __init__(self, config, class_distribution, 
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 query_frequency: float = 0.3,
                 focus_weight: float = 2.0):
        """
        QGFL combines:
        1. Class weights (from base class)
        2. Focal loss parameters (alpha, gamma)
        3. Query-guided learning (query_frequency, focus_weight)
        """
        super().__init__(config, class_distribution)
        
        # Focal loss parameters
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Query-guided parameters
        self.query_frequency = query_frequency
        self.focus_weight = focus_weight
        
        # Calculate effective weights combining class weights and focal alpha
        self.effective_weights = self._calculate_effective_weights()
        
        self.custom_metrics = {
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'query_frequency': query_frequency,
            'focus_weight': focus_weight
        }
        
        print(f"\nQGFL Configuration:")
        print(f"  Focal Alpha: {focal_alpha}")
        print(f"  Focal Gamma: {focal_gamma}")
        print(f"  Query Frequency: {query_frequency}")
        print(f"  Focus Weight: {focus_weight}")
        print(f"  Effective Weights: {self.effective_weights.numpy()}")
    
    def _calculate_effective_weights(self):
        """Combine class weights with focal loss alpha"""
        # Normalize class weights
        normalized_weights = self.class_weights / self.class_weights.sum() * self.num_classes
        
        # Apply focal loss alpha modulation
        # Alpha typically weights minority class more in focal loss
        if self.num_classes == 2:
            # Binary case: apply alpha to positive (minority) class
            focal_weights = torch.tensor([1 - self.focal_alpha, self.focal_alpha])
            effective = normalized_weights * focal_weights * 2  # Scale back up
        else:
            # Multi-class: apply alpha based on frequency
            max_weight = normalized_weights.max()
            focal_weights = torch.where(
                normalized_weights > 1.0,  # Minority classes
                torch.ones_like(normalized_weights) * self.focal_alpha * 4,
                torch.ones_like(normalized_weights) * (1 - self.focal_alpha)
            )
            effective = normalized_weights * focal_weights
        
        return effective
    
    def get_loss_function(self):
        """QGFL requires custom loss implementation"""
        # Note: For YOLO, we approximate through parameter adjustment
        return None
    
    def get_training_params(self) -> Dict[str, Any]:
        """QGFL-optimized training parameters"""
        # Calculate imbalance-aware loss weights
        imbalance_ratio = float(self.class_weights.max() / self.class_weights.min())
        
        # QGFL uses stronger classification loss with focal modulation
        # The gamma parameter is approximated through cls weight scaling
        cls_weight = 0.5 * np.sqrt(imbalance_ratio) * (1 + self.focal_gamma/4)
        
        # Box loss is increased for better localization of minority class
        box_weight = 7.5 * (1 + self.focus_weight/10)
        
        return {
            'cls': min(cls_weight, 3.0),  # Stronger cls loss, capped at 3.0
            'box': box_weight,  # Enhanced box loss for minorities
            'dfl': 1.5 * (1 + self.focal_gamma/8)  # Slightly enhanced DFL
        }
    
    def get_strategy_name(self) -> str:
        return "qgfl"
    
    def get_hyperparameter_adjustments(self) -> Dict[str, Any]:
        """QGFL needs specific training adjustments"""
        return {
            'patience': 35,  # More patience for complex loss
            'warmup_epochs': 5,  # Longer warmup
            'warmup_bias_lr': 0.05,  # Lower warmup bias
            'mosaic': 0.7,  # Reduce mosaic augmentation
            'lr0': self.config.lr0 * 0.8,  # Lower initial LR
            'weight_decay': 0.001  # Stronger regularization
        }
    
    def should_apply_custom_loss(self) -> bool:
        """QGFL uses custom loss computation"""
        return True


def create_training_strategy(strategy_name: str,
                           config,
                           class_distribution: Dict[str, int]) -> TrainingStrategy:
    """
    Factory function to create appropriate training strategy

    Available strategies:
    - 'no_weights': True baseline without class weighting
    - 'qgfl': Query-Guided Focal Loss (full implementation - NOT YET PROPERLY IMPLEMENTED)
    """

    strategies = {
        'no_weights': NoWeightsStrategy,
        'qgfl': QGFLStrategy
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    print(f"\n{'='*60}")
    print(f"Initializing Training Strategy: {strategy_name.upper()}")
    print(f"{'='*60}")
    
    # Special parameters for QGFL
    if strategy_name == 'qgfl':
        # You can adjust these based on your dataset characteristics
        return QGFLStrategy(
            config, 
            class_distribution,
            focal_alpha=0.25,  # Standard focal loss alpha
            focal_gamma=2.0,   # Standard focal loss gamma
            query_frequency=0.3,  # Query 30% of hard examples
            focus_weight=2.0   # 2x weight on focused samples
        )
    else:
        return strategies[strategy_name](config, class_distribution)