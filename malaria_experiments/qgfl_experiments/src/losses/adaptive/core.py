"""
Dynamic Adaptive QGFL Core Implementation

Extends QGFLCore with per-class parameter tracking and epoch-level adaptation.
Automatically adjusts alpha, gamma, and threshold based on validation metrics.

Author: Thabang Isaka
Date: 2025-11-21
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# Import base QGFLCore
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.losses.qgfl_core import QGFLCore
from src.losses.adaptive.config import AdaptiveQGFLConfig


class DynamicAdaptiveQGFLCore(QGFLCore):
    """
    Dynamic Adaptive Quality-Guided Focal Loss

    Extends QGFLCore with:
    1. Per-class alpha and gamma parameters (works for any nc)
    2. Epoch-level adaptation based on validation metrics
    3. History tracking for visualization
    4. Warmup and freeze-final-epochs logic

    Adaptation Rules:
    - Low recall on minority class → increase alpha, increase gamma
    - Low precision on minority class → decrease alpha, decrease gamma
    - Target: maximize F1 while maintaining recall >= target_recall
    """

    def __init__(self, config: AdaptiveQGFLConfig):
        """
        Initialize Dynamic Adaptive QGFL

        Args:
            config: AdaptiveQGFLConfig with task/dataset/model settings
        """
        # Initialize base class with config values
        super().__init__(
            infected_alpha=config.initial_alpha,
            uninfected_alpha=1.0 - config.initial_alpha,
            infected_gamma=config.initial_gamma,
            uninfected_gamma=2.0,  # Standard focal loss for majority
            difficulty_threshold=config.initial_threshold,
            quality_margin=config.quality_margin,
            quality_factor=config.quality_factor,
            uiou_start=config.uiou_start,
            uiou_end=config.uiou_end,
            epsilon=1e-7,
            debug=config.debug
        )

        self.config = config

        # === PER-CLASS PARAMETERS ===
        # These are tracked per-class and adapted during training
        self.class_alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(config.initial_alpha if i > 0 else 1.0 - config.initial_alpha))
            for i in range(config.nc)
        ])

        self.class_gammas = nn.ParameterList([
            nn.Parameter(torch.tensor(config.initial_gamma if i > 0 else 2.0))
            for i in range(config.nc)
        ])

        # Current threshold (shared across classes)
        self.current_threshold = config.initial_threshold

        # === ADAPTATION STATE ===
        self.current_epoch = 0
        self.total_epochs = 200  # Will be set during training
        self.is_adapting = False  # True during adaptation window

        # === EXPONENTIAL MOVING AVERAGES ===
        # Smoothed metrics for stable adaptation
        self.ema_recall = {i: 0.5 for i in range(config.nc)}
        self.ema_precision = {i: 0.5 for i in range(config.nc)}
        self.ema_f1 = {i: 0.5 for i in range(config.nc)}

        # === HISTORY TRACKING ===
        self.history = {
            'epoch': [],
            'alphas': defaultdict(list),
            'gammas': defaultdict(list),
            'threshold': [],
            'recall': defaultdict(list),
            'precision': defaultdict(list),
            'f1': defaultdict(list),
        }

        if config.debug:
            self._print_init_info()

    def _print_init_info(self):
        """Print initialization information"""
        print("\n" + "=" * 60)
        print("Dynamic Adaptive QGFL Initialized")
        print("=" * 60)
        print(f"Task: {self.config.task} (nc={self.config.nc})")
        print(f"Dataset: {self.config.dataset}")
        print(f"Model: {self.config.model_type}")
        print(f"\nInitial Parameters:")
        for i in range(self.config.nc):
            class_name = self.config.class_names.get(i, f"Class_{i}")
            print(f"  {class_name}: α={self.class_alphas[i].item():.3f}, γ={self.class_gammas[i].item():.1f}")
        print(f"  Threshold: {self.current_threshold:.3f}")
        print(f"\nAdaptation Settings:")
        print(f"  Rate: {self.config.adaptation_rate}")
        print(f"  Momentum: {self.config.adaptation_momentum}")
        print(f"  Warmup: {self.config.warmup_epochs} epochs")
        print(f"  Freeze Final: {self.config.freeze_final_epochs} epochs")
        print(f"  Target Recall: {self.config.target_recall}")
        print(f"  Min Precision: {self.config.min_precision}")
        print("=" * 60 + "\n")

    def should_adapt(self, epoch: int, total_epochs: int) -> bool:
        """
        Check if adaptation should occur this epoch

        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total training epochs

        Returns:
            True if we should adapt parameters
        """
        # No adaptation during warmup
        if epoch < self.config.warmup_epochs:
            return False

        # No adaptation in final epochs (freeze for stability)
        if epoch >= total_epochs - self.config.freeze_final_epochs:
            return False

        return True

    def adapt_parameters(
        self,
        epoch: int,
        metrics: Dict[str, Dict[int, float]],
        total_epochs: int
    ) -> Dict[str, any]:
        """
        Adapt parameters based on validation metrics

        This is the core adaptation logic called after each validation epoch.

        Args:
            epoch: Current epoch (0-indexed)
            metrics: Dict with keys 'recall', 'precision', 'f1'
                    Each maps class_id -> float value
            total_epochs: Total training epochs

        Returns:
            Dict of adaptation info for logging
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        adaptation_info = {
            'epoch': epoch,
            'adapted': False,
            'changes': {}
        }

        # Update EMAs first
        self._update_ema(metrics)

        # Check if we should adapt
        if not self.should_adapt(epoch, total_epochs):
            reason = "warmup" if epoch < self.config.warmup_epochs else "freeze"
            adaptation_info['skip_reason'] = reason
            self._record_history(epoch, metrics)
            return adaptation_info

        self.is_adapting = True
        adaptation_info['adapted'] = True

        # Adapt each minority class
        for class_id in self.config.get_minority_class_ids():
            class_name = self.config.class_names.get(class_id, f"Class_{class_id}")

            # Get smoothed metrics
            recall = self.ema_recall[class_id]
            precision = self.ema_precision[class_id]
            f1 = self.ema_f1[class_id]

            # Compute parameter adjustments
            alpha_delta, gamma_delta = self._compute_adjustments(
                recall, precision, class_id
            )

            # Apply adjustments with bounds checking
            old_alpha = self.class_alphas[class_id].item()
            old_gamma = self.class_gammas[class_id].item()

            new_alpha = self._apply_alpha_adjustment(class_id, alpha_delta)
            new_gamma = self._apply_gamma_adjustment(class_id, gamma_delta)

            adaptation_info['changes'][class_name] = {
                'alpha': {'old': old_alpha, 'new': new_alpha, 'delta': alpha_delta},
                'gamma': {'old': old_gamma, 'new': new_gamma, 'delta': gamma_delta},
                'recall': recall,
                'precision': precision,
                'f1': f1
            }

        # Adapt threshold (shared across classes)
        threshold_delta = self._compute_threshold_adjustment()
        old_threshold = self.current_threshold
        self.current_threshold = self._apply_threshold_adjustment(threshold_delta)

        adaptation_info['threshold'] = {
            'old': old_threshold,
            'new': self.current_threshold,
            'delta': threshold_delta
        }

        # Record history
        self._record_history(epoch, metrics)

        # Update base class parameters for compatibility
        self._sync_to_base_class()

        if self.config.debug:
            self._print_adaptation_summary(adaptation_info)

        return adaptation_info

    def _update_ema(self, metrics: Dict[str, Dict[int, float]]):
        """Update exponential moving averages of metrics"""
        momentum = self.config.adaptation_momentum

        for class_id in range(self.config.nc):
            if class_id in metrics.get('recall', {}):
                self.ema_recall[class_id] = (
                    momentum * self.ema_recall[class_id] +
                    (1 - momentum) * metrics['recall'][class_id]
                )
            if class_id in metrics.get('precision', {}):
                self.ema_precision[class_id] = (
                    momentum * self.ema_precision[class_id] +
                    (1 - momentum) * metrics['precision'][class_id]
                )
            if class_id in metrics.get('f1', {}):
                self.ema_f1[class_id] = (
                    momentum * self.ema_f1[class_id] +
                    (1 - momentum) * metrics['f1'][class_id]
                )

    def _compute_adjustments(
        self,
        recall: float,
        precision: float,
        class_id: int
    ) -> Tuple[float, float]:
        """
        Compute alpha and gamma adjustments based on recall/precision

        Adaptation Rules:
        - recall < target_recall → increase alpha, increase gamma (focus more on class)
        - precision < min_precision → decrease alpha, decrease gamma (reduce over-prediction)
        - recall >= target AND precision >= min → small adjustments toward balance

        Args:
            recall: Current recall for this class
            precision: Current precision for this class
            class_id: Class ID

        Returns:
            (alpha_delta, gamma_delta) adjustments
        """
        rate = self.config.adaptation_rate
        target_recall = self.config.target_recall
        min_precision = self.config.min_precision

        alpha_delta = 0.0
        gamma_delta = 0.0

        # Recall deficit: need to focus more on this class
        if recall < target_recall:
            recall_gap = target_recall - recall
            # Proportional increase: bigger gap → bigger increase
            alpha_delta += rate * recall_gap * 2.0
            gamma_delta += rate * recall_gap * 10.0  # Gamma moves faster

        # Precision deficit: over-predicting, need to reduce focus
        if precision < min_precision:
            precision_gap = min_precision - precision
            # Reduce alpha and gamma
            alpha_delta -= rate * precision_gap * 1.5
            gamma_delta -= rate * precision_gap * 8.0

        # If both are good, fine-tune toward F1 balance
        if recall >= target_recall and precision >= min_precision:
            f1 = self.ema_f1[class_id]

            # If recall >> precision, slightly reduce alpha
            if recall > precision + 0.1:
                alpha_delta -= rate * 0.5
                gamma_delta -= rate * 2.0
            # If precision >> recall, slightly increase alpha
            elif precision > recall + 0.1:
                alpha_delta += rate * 0.5
                gamma_delta += rate * 2.0

        return alpha_delta, gamma_delta

    def _apply_alpha_adjustment(self, class_id: int, delta: float) -> float:
        """Apply alpha adjustment with bounds checking"""
        current = self.class_alphas[class_id].item()
        new_value = current + delta

        # Clamp to bounds
        new_value = max(self.config.alpha_bounds[0],
                       min(self.config.alpha_bounds[1], new_value))

        # Update parameter (requires grad=False for safety)
        with torch.no_grad():
            self.class_alphas[class_id].fill_(new_value)

        return new_value

    def _apply_gamma_adjustment(self, class_id: int, delta: float) -> float:
        """Apply gamma adjustment with bounds checking"""
        current = self.class_gammas[class_id].item()
        new_value = current + delta

        # Clamp to bounds
        new_value = max(self.config.gamma_bounds[0],
                       min(self.config.gamma_bounds[1], new_value))

        # Update parameter
        with torch.no_grad():
            self.class_gammas[class_id].fill_(new_value)

        return new_value

    def _compute_threshold_adjustment(self) -> float:
        """
        Compute threshold adjustment based on overall minority performance

        If minorities have good recall but poor precision → increase threshold
        If minorities have poor recall → decrease threshold
        """
        rate = self.config.adaptation_rate

        # Average metrics across minority classes
        minority_ids = self.config.get_minority_class_ids()
        if not minority_ids:
            return 0.0

        avg_recall = np.mean([self.ema_recall[i] for i in minority_ids])
        avg_precision = np.mean([self.ema_precision[i] for i in minority_ids])

        # Threshold affects difficulty classification
        # Higher threshold = more samples treated as "difficult" = stronger focus
        if avg_recall < self.config.target_recall:
            # Need more focus → lower threshold (more samples get boosted)
            return -rate * 0.5
        elif avg_precision < self.config.min_precision:
            # Too many false positives → higher threshold (less aggressive boosting)
            return rate * 0.5

        return 0.0

    def _apply_threshold_adjustment(self, delta: float) -> float:
        """Apply threshold adjustment with bounds checking"""
        new_value = self.current_threshold + delta

        # Clamp to bounds
        new_value = max(self.config.threshold_bounds[0],
                       min(self.config.threshold_bounds[1], new_value))

        return new_value

    def _sync_to_base_class(self):
        """Sync adapted parameters back to base QGFLCore attributes"""
        # For binary classification (nc=2)
        if self.config.nc == 2:
            self.infected_alpha = self.class_alphas[1].item()
            self.uninfected_alpha = self.class_alphas[0].item()
            self.infected_gamma = self.class_gammas[1].item()
            self.uninfected_gamma = self.class_gammas[0].item()
        else:
            # For multiclass, use average of minority class values
            minority_ids = self.config.get_minority_class_ids()
            self.infected_alpha = np.mean([
                self.class_alphas[i].item() for i in minority_ids
            ])
            self.infected_gamma = np.mean([
                self.class_gammas[i].item() for i in minority_ids
            ])
            self.uninfected_alpha = self.class_alphas[0].item()
            self.uninfected_gamma = self.class_gammas[0].item()

        self.difficulty_threshold = self.current_threshold

    def _record_history(self, epoch: int, metrics: Dict):
        """Record current state to history for visualization"""
        self.history['epoch'].append(epoch)
        self.history['threshold'].append(self.current_threshold)

        for class_id in range(self.config.nc):
            self.history['alphas'][class_id].append(
                self.class_alphas[class_id].item()
            )
            self.history['gammas'][class_id].append(
                self.class_gammas[class_id].item()
            )
            self.history['recall'][class_id].append(
                metrics.get('recall', {}).get(class_id, 0.0)
            )
            self.history['precision'][class_id].append(
                metrics.get('precision', {}).get(class_id, 0.0)
            )
            self.history['f1'][class_id].append(
                metrics.get('f1', {}).get(class_id, 0.0)
            )

    def _print_adaptation_summary(self, info: Dict):
        """Print adaptation summary for debugging"""
        print(f"\n[Adaptive QGFL] Epoch {info['epoch']} Adaptation Summary:")

        if not info['adapted']:
            print(f"  Skipped: {info.get('skip_reason', 'unknown')}")
            return

        for class_name, changes in info.get('changes', {}).items():
            print(f"  {class_name}:")
            print(f"    α: {changes['alpha']['old']:.3f} → {changes['alpha']['new']:.3f} "
                  f"(Δ={changes['alpha']['delta']:+.4f})")
            print(f"    γ: {changes['gamma']['old']:.1f} → {changes['gamma']['new']:.1f} "
                  f"(Δ={changes['gamma']['delta']:+.2f})")
            print(f"    Metrics: R={changes['recall']:.3f}, P={changes['precision']:.3f}, "
                  f"F1={changes['f1']:.3f}")

        if 'threshold' in info:
            t = info['threshold']
            print(f"  Threshold: {t['old']:.3f} → {t['new']:.3f} (Δ={t['delta']:+.4f})")

    def get_class_alpha_tensor(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Get alpha values for given class IDs

        Args:
            class_ids: Tensor of class IDs [N]

        Returns:
            Alpha values for each sample [N]
        """
        device = class_ids.device
        alphas = torch.zeros_like(class_ids, dtype=torch.float32)

        for i in range(self.config.nc):
            mask = class_ids == i
            alphas[mask] = self.class_alphas[i].item()

        return alphas.to(device)

    def get_class_gamma_tensor(self, class_ids: torch.Tensor) -> torch.Tensor:
        """
        Get gamma values for given class IDs

        Args:
            class_ids: Tensor of class IDs [N]

        Returns:
            Gamma values for each sample [N]
        """
        device = class_ids.device
        gammas = torch.zeros_like(class_ids, dtype=torch.float32)

        for i in range(self.config.nc):
            mask = class_ids == i
            gammas[mask] = self.class_gammas[i].item()

        return gammas.to(device)

    def get_current_params_dict(self) -> Dict:
        """Get current parameters as dictionary for W&B logging"""
        params = {
            'epoch': self.current_epoch,
            'threshold': self.current_threshold,
            'is_adapting': self.is_adapting,
        }

        for i in range(self.config.nc):
            class_name = self.config.class_names.get(i, f"class_{i}")
            params[f'alpha_{class_name}'] = self.class_alphas[i].item()
            params[f'gamma_{class_name}'] = self.class_gammas[i].item()

        return params

    def get_history_dataframe(self):
        """Convert history to pandas DataFrame for analysis"""
        try:
            import pandas as pd

            data = {'epoch': self.history['epoch']}
            data['threshold'] = self.history['threshold']

            for class_id in range(self.config.nc):
                class_name = self.config.class_names.get(class_id, f"class_{class_id}")
                data[f'alpha_{class_name}'] = self.history['alphas'][class_id]
                data[f'gamma_{class_name}'] = self.history['gammas'][class_id]
                data[f'recall_{class_name}'] = self.history['recall'][class_id]
                data[f'precision_{class_name}'] = self.history['precision'][class_id]
                data[f'f1_{class_name}'] = self.history['f1'][class_id]

            return pd.DataFrame(data)
        except ImportError:
            return self.history


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DynamicAdaptiveQGFLCore")
    print("=" * 60)

    # Test 1: Basic initialization
    print("\nTest 1: Basic initialization (binary task)")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1', debug=False)
    core = DynamicAdaptiveQGFLCore(config)

    assert len(core.class_alphas) == 2, f"Expected 2 class alphas, got {len(core.class_alphas)}"
    assert len(core.class_gammas) == 2, f"Expected 2 class gammas, got {len(core.class_gammas)}"
    print(f"  Class alphas: {[p.item() for p in core.class_alphas]}")
    print(f"  Class gammas: {[p.item() for p in core.class_gammas]}")
    print("  PASSED")

    # Test 2: Multiclass initialization (staging)
    print("\nTest 2: Multiclass initialization (staging task)")
    config = AdaptiveQGFLConfig(task='staging', dataset='d1', debug=False)
    core = DynamicAdaptiveQGFLCore(config)

    assert len(core.class_alphas) == 5, f"Expected 5 class alphas, got {len(core.class_alphas)}"
    assert len(core.class_gammas) == 5, f"Expected 5 class gammas, got {len(core.class_gammas)}"
    print(f"  nc={config.nc}, class_alphas count={len(core.class_alphas)}")
    print("  PASSED")

    # Test 3: should_adapt logic
    print("\nTest 3: should_adapt logic")
    config = AdaptiveQGFLConfig(
        task='binary',
        warmup_epochs=5,
        freeze_final_epochs=20,
        debug=False
    )
    core = DynamicAdaptiveQGFLCore(config)

    # During warmup
    assert core.should_adapt(0, 200) == False, "Should not adapt during warmup"
    assert core.should_adapt(4, 200) == False, "Should not adapt during warmup"

    # After warmup, before freeze
    assert core.should_adapt(5, 200) == True, "Should adapt after warmup"
    assert core.should_adapt(100, 200) == True, "Should adapt mid-training"
    assert core.should_adapt(179, 200) == True, "Should adapt before freeze"

    # During freeze
    assert core.should_adapt(180, 200) == False, "Should not adapt during freeze"
    assert core.should_adapt(199, 200) == False, "Should not adapt during freeze"

    print("  Warmup (0-4): adapt=False ✓")
    print("  Mid-training (5-179): adapt=True ✓")
    print("  Freeze (180-199): adapt=False ✓")
    print("  PASSED")

    # Test 4: Adaptation with low recall
    print("\nTest 4: Adaptation with low recall (should increase alpha/gamma)")
    config = AdaptiveQGFLConfig(
        task='binary',
        dataset='d1',
        warmup_epochs=0,
        freeze_final_epochs=0,
        target_recall=0.90,
        adaptation_rate=0.1,
        adaptation_momentum=0.0,  # No momentum for cleaner testing
        debug=False
    )
    core = DynamicAdaptiveQGFLCore(config)

    initial_alpha = core.class_alphas[1].item()
    initial_gamma = core.class_gammas[1].item()

    # Simulate low recall on infected class
    metrics = {
        'recall': {0: 0.95, 1: 0.50},  # Low recall on minority
        'precision': {0: 0.80, 1: 0.70},
        'f1': {0: 0.87, 1: 0.58}
    }

    info = core.adapt_parameters(epoch=10, metrics=metrics, total_epochs=200)

    new_alpha = core.class_alphas[1].item()
    new_gamma = core.class_gammas[1].item()

    assert new_alpha > initial_alpha, f"Alpha should increase: {initial_alpha:.3f} -> {new_alpha:.3f}"
    assert new_gamma > initial_gamma, f"Gamma should increase: {initial_gamma:.1f} -> {new_gamma:.1f}"
    print(f"  Initial: α={initial_alpha:.3f}, γ={initial_gamma:.1f}")
    print(f"  After adaptation: α={new_alpha:.3f}, γ={new_gamma:.1f}")
    print(f"  Changes applied: ✓")
    print("  PASSED")

    # Test 5: Adaptation with low precision
    print("\nTest 5: Adaptation with low precision (should decrease alpha/gamma)")
    config = AdaptiveQGFLConfig(
        task='binary',
        dataset='d1',
        warmup_epochs=0,
        freeze_final_epochs=0,
        min_precision=0.60,
        target_recall=0.90,
        adaptation_rate=0.1,
        adaptation_momentum=0.0,  # No momentum for cleaner testing
        debug=False
    )
    core = DynamicAdaptiveQGFLCore(config)

    # Set high initial values
    with torch.no_grad():
        core.class_alphas[1].fill_(0.9)
        core.class_gammas[1].fill_(8.0)

    initial_alpha = core.class_alphas[1].item()
    initial_gamma = core.class_gammas[1].item()

    # Simulate high recall but low precision on infected class
    # This should cause alpha/gamma to decrease (over-predicting)
    metrics = {
        'recall': {0: 0.70, 1: 0.95},  # High recall (above target 0.90)
        'precision': {0: 0.90, 1: 0.30},  # Low precision on minority
        'f1': {0: 0.79, 1: 0.46}
    }

    info = core.adapt_parameters(epoch=10, metrics=metrics, total_epochs=200)

    new_alpha = core.class_alphas[1].item()
    new_gamma = core.class_gammas[1].item()

    assert new_alpha < initial_alpha, f"Alpha should decrease: {initial_alpha:.3f} -> {new_alpha:.3f}"
    assert new_gamma < initial_gamma, f"Gamma should decrease: {initial_gamma:.1f} -> {new_gamma:.1f}"
    print(f"  Initial: α={initial_alpha:.3f}, γ={initial_gamma:.1f}")
    print(f"  After adaptation: α={new_alpha:.3f}, γ={new_gamma:.1f}")
    print(f"  Changes applied: ✓")
    print("  PASSED")

    # Test 6: Bounds checking
    print("\nTest 6: Parameter bounds checking")
    config = AdaptiveQGFLConfig(
        task='binary',
        alpha_bounds=(0.3, 0.95),
        gamma_bounds=(2.0, 10.0),
        debug=False
    )
    core = DynamicAdaptiveQGFLCore(config)

    # Try to exceed upper bound
    with torch.no_grad():
        core.class_alphas[1].fill_(0.94)
    new_alpha = core._apply_alpha_adjustment(1, 0.10)  # Try to add 0.10
    assert new_alpha <= 0.95, f"Alpha should be capped at 0.95, got {new_alpha}"

    # Try to go below lower bound
    with torch.no_grad():
        core.class_alphas[1].fill_(0.32)
    new_alpha = core._apply_alpha_adjustment(1, -0.10)  # Try to subtract 0.10
    assert new_alpha >= 0.30, f"Alpha should be floored at 0.30, got {new_alpha}"

    print(f"  Upper bound (0.95) respected: ✓")
    print(f"  Lower bound (0.30) respected: ✓")
    print("  PASSED")

    # Test 7: get_class_alpha_tensor
    print("\nTest 7: get_class_alpha_tensor")
    config = AdaptiveQGFLConfig(task='binary', debug=False)
    core = DynamicAdaptiveQGFLCore(config)

    with torch.no_grad():
        core.class_alphas[0].fill_(0.35)  # Majority
        core.class_alphas[1].fill_(0.65)  # Minority

    class_ids = torch.tensor([0, 1, 1, 0, 1])
    alphas = core.get_class_alpha_tensor(class_ids)

    expected = torch.tensor([0.35, 0.65, 0.65, 0.35, 0.65])
    assert torch.allclose(alphas, expected), f"Expected {expected}, got {alphas}"
    print(f"  class_ids: {class_ids.tolist()}")
    print(f"  alphas: {alphas.tolist()}")
    print("  PASSED")

    # Test 8: History tracking
    print("\nTest 8: History tracking")
    config = AdaptiveQGFLConfig(task='binary', warmup_epochs=0, debug=False)
    core = DynamicAdaptiveQGFLCore(config)

    for epoch in range(5):
        metrics = {
            'recall': {0: 0.8 + epoch*0.02, 1: 0.5 + epoch*0.05},
            'precision': {0: 0.75, 1: 0.6},
            'f1': {0: 0.77, 1: 0.55}
        }
        core.adapt_parameters(epoch, metrics, 200)

    assert len(core.history['epoch']) == 5, f"Expected 5 history entries"
    assert len(core.history['alphas'][1]) == 5, f"Expected 5 alpha history entries"
    print(f"  Epochs recorded: {core.history['epoch']}")
    print(f"  Alpha[1] history: {[f'{x:.3f}' for x in core.history['alphas'][1]]}")
    print("  PASSED")

    # Test 9: get_current_params_dict for W&B
    print("\nTest 9: get_current_params_dict for W&B logging")
    config = AdaptiveQGFLConfig(task='binary', debug=False)
    core = DynamicAdaptiveQGFLCore(config)
    core.current_epoch = 50

    params = core.get_current_params_dict()
    assert 'epoch' in params
    assert 'alpha_Uninfected' in params
    assert 'alpha_Infected' in params
    assert 'gamma_Uninfected' in params
    assert 'gamma_Infected' in params
    print(f"  Keys: {list(params.keys())}")
    print("  PASSED")

    # Test 10: Species task (5 classes)
    print("\nTest 10: Species task with 5 classes")
    config = AdaptiveQGFLConfig(task='species', dataset='d3', debug=False)
    core = DynamicAdaptiveQGFLCore(config)

    assert len(core.class_alphas) == 5
    minority_ids = config.get_minority_class_ids()
    assert minority_ids == [1, 2, 3, 4], f"Expected [1,2,3,4], got {minority_ids}"

    # Check all minority classes have initial_alpha (with tolerance for float comparison)
    for i in minority_ids:
        actual = core.class_alphas[i].item()
        expected = config.initial_alpha
        assert abs(actual - expected) < 1e-5, f"Class {i}: expected {expected}, got {actual}"

    print(f"  Minority class IDs: {minority_ids}")
    print(f"  All minority alphas = {config.initial_alpha}: ✓")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL CORE TESTS PASSED")
    print("=" * 60)
