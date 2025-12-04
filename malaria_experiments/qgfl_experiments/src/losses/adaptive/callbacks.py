"""
Adaptive Training Callbacks for Dynamic QGFL

Provides callback functions to integrate adaptation with YOLO/RT-DETR training loops.
Callbacks trigger adaptation after validation epochs based on per-class metrics.

Author: Thabang Isaka
Date: 2025-11-21
"""

import torch
from typing import Dict, Optional, Union, Any
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class AdaptiveTrainingCallback:
    """
    Callback handler for Dynamic Adaptive QGFL training

    Integrates with YOLO/RT-DETR training loops to:
    1. Extract per-class metrics after validation
    2. Trigger parameter adaptation
    3. Log adaptation changes to W&B

    Usage:
        callback = AdaptiveTrainingCallback(loss_fn, config)
        # In training loop:
        callback.on_fit_epoch_end(trainer)  # After each epoch
    """

    def __init__(
        self,
        loss_fn: Any,  # DynamicAdaptiveYOLOLoss or DynamicAdaptiveRTDETRLoss
        config: Any,   # AdaptiveQGFLConfig
        use_wandb: bool = True
    ):
        """
        Initialize callback

        Args:
            loss_fn: The adaptive loss function instance
            config: AdaptiveQGFLConfig with task/model settings
            use_wandb: Whether to log to Weights & Biases
        """
        self.loss_fn = loss_fn
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB

        self.epoch = 0
        self.total_epochs = 200

        # Track metrics for logging
        self.metrics_history = defaultdict(list)

        if config.debug:
            print(f"\n[AdaptiveCallback] Initialized")
            print(f"  Use W&B: {self.use_wandb}")

    def on_train_start(self, trainer):
        """
        Called at the start of training

        Args:
            trainer: YOLO/RT-DETR trainer object
        """
        self.total_epochs = trainer.epochs

        if self.use_wandb:
            # Log initial configuration
            wandb.config.update({
                'adaptive_qgfl': self.config.to_dict()
            }, allow_val_change=True)

    def on_fit_epoch_end(self, trainer):
        """
        Called after each training + validation epoch

        This is the main adaptation trigger. Extracts validation metrics
        and calls adapt_parameters() on the loss function.

        Args:
            trainer: YOLO/RT-DETR trainer object with metrics attribute
        """
        self.epoch = trainer.epoch

        # Extract per-class metrics from trainer
        metrics = self._extract_metrics(trainer)

        if metrics is None:
            if self.config.debug:
                print(f"[AdaptiveCallback] Epoch {self.epoch}: No metrics available yet")
            return

        # Trigger adaptation
        adaptation_info = self.loss_fn.adapt_parameters(
            epoch=self.epoch,
            metrics=metrics,
            total_epochs=self.total_epochs
        )

        # Log to W&B
        if self.use_wandb:
            self._log_to_wandb(adaptation_info, metrics)

        return adaptation_info

    def _extract_metrics(self, trainer) -> Optional[Dict]:
        """
        Extract per-class recall, precision, F1 from trainer metrics

        Args:
            trainer: YOLO/RT-DETR trainer with metrics attribute

        Returns:
            Dict with 'recall', 'precision', 'f1' keys, each mapping
            class_id -> float value
        """
        if not hasattr(trainer, 'metrics'):
            return None

        metrics = trainer.metrics
        if metrics is None:
            return None

        result = {
            'recall': {},
            'precision': {},
            'f1': {}
        }

        # Try different attribute patterns (YOLO vs RT-DETR may differ)
        # Pattern 1: metrics.box.per_class_results
        if hasattr(metrics, 'box') and hasattr(metrics.box, 'r'):
            # YOLO v8/v11 pattern
            try:
                r = metrics.box.r  # Per-class recall
                p = metrics.box.p  # Per-class precision

                for class_id in range(self.config.nc):
                    if class_id < len(r):
                        result['recall'][class_id] = float(r[class_id])
                        result['precision'][class_id] = float(p[class_id])
                        # Compute F1
                        if r[class_id] + p[class_id] > 0:
                            f1 = 2 * r[class_id] * p[class_id] / (r[class_id] + p[class_id])
                        else:
                            f1 = 0.0
                        result['f1'][class_id] = float(f1)
                return result
            except Exception as e:
                if self.config.debug:
                    print(f"[AdaptiveCallback] Error extracting box metrics: {e}")

        # Pattern 2: metrics.results_dict
        if hasattr(metrics, 'results_dict'):
            try:
                rd = metrics.results_dict

                for class_id in range(self.config.nc):
                    # Try per-class keys
                    r_key = f'metrics/recall(B)_{class_id}'
                    p_key = f'metrics/precision(B)_{class_id}'

                    if r_key in rd:
                        result['recall'][class_id] = float(rd[r_key])
                        result['precision'][class_id] = float(rd[p_key])
                    else:
                        # Fallback to overall metrics
                        if 'metrics/recall(B)' in rd:
                            result['recall'][class_id] = float(rd['metrics/recall(B)'])
                            result['precision'][class_id] = float(rd['metrics/precision(B)'])

                    # Compute F1 from R and P
                    r = result['recall'].get(class_id, 0)
                    p = result['precision'].get(class_id, 0)
                    if r + p > 0:
                        result['f1'][class_id] = 2 * r * p / (r + p)
                    else:
                        result['f1'][class_id] = 0.0

                if result['recall']:
                    return result
            except Exception as e:
                if self.config.debug:
                    print(f"[AdaptiveCallback] Error extracting results_dict: {e}")

        # Pattern 3: Direct per_class_results attribute
        if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
            try:
                val_metrics = trainer.validator.metrics
                if hasattr(val_metrics, 'ap_class_index'):
                    # This gives us per-class metrics
                    pass  # Not all versions have this
            except:
                pass

        return None if not result['recall'] else result

    def _log_to_wandb(self, adaptation_info: Dict, metrics: Dict):
        """
        Log adaptation info and metrics to Weights & Biases

        Args:
            adaptation_info: Dict from adapt_parameters()
            metrics: Extracted per-class metrics
        """
        if not self.use_wandb:
            return

        log_dict = {
            'adaptive_qgfl/epoch': self.epoch,
            'adaptive_qgfl/adapted': adaptation_info.get('adapted', False),
            'adaptive_qgfl/threshold': self.loss_fn.current_threshold,
        }

        # Log per-class parameters
        for class_id in range(self.config.nc):
            class_name = self.config.class_names.get(class_id, f"class_{class_id}")
            log_dict[f'adaptive_qgfl/alpha_{class_name}'] = self.loss_fn.class_alphas[class_id].item()
            log_dict[f'adaptive_qgfl/gamma_{class_name}'] = self.loss_fn.class_gammas[class_id].item()

            if class_id in metrics.get('recall', {}):
                log_dict[f'adaptive_qgfl/recall_{class_name}'] = metrics['recall'][class_id]
                log_dict[f'adaptive_qgfl/precision_{class_name}'] = metrics['precision'][class_id]
                log_dict[f'adaptive_qgfl/f1_{class_name}'] = metrics['f1'][class_id]

        # Log changes if adaptation occurred
        if adaptation_info.get('adapted'):
            for class_name, changes in adaptation_info.get('changes', {}).items():
                log_dict[f'adaptive_qgfl/alpha_delta_{class_name}'] = changes['alpha']['delta']
                log_dict[f'adaptive_qgfl/gamma_delta_{class_name}'] = changes['gamma']['delta']

        try:
            wandb.log(log_dict)
        except Exception as e:
            if self.config.debug:
                print(f"[AdaptiveCallback] W&B logging failed: {e}")

    def get_callback_dict(self) -> Dict[str, callable]:
        """
        Return callback dict for YOLO trainer.add_callback()

        Returns:
            Dict mapping callback names to methods
        """
        return {
            'on_train_start': self.on_train_start,
            'on_fit_epoch_end': self.on_fit_epoch_end,
        }


def create_adaptive_callback(
    loss_fn,
    config,
    use_wandb: bool = True
) -> AdaptiveTrainingCallback:
    """
    Factory function to create adaptive callback

    Args:
        loss_fn: DynamicAdaptiveYOLOLoss or DynamicAdaptiveRTDETRLoss
        config: AdaptiveQGFLConfig
        use_wandb: Whether to enable W&B logging

    Returns:
        AdaptiveTrainingCallback instance
    """
    return AdaptiveTrainingCallback(loss_fn, config, use_wandb)


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    from src.losses.adaptive.config import AdaptiveQGFLConfig
    from src.losses.adaptive.yolo_adapter import DynamicAdaptiveYOLOLoss

    print("=" * 60)
    print("Testing AdaptiveTrainingCallback")
    print("=" * 60)

    # Test 1: Basic initialization
    print("\nTest 1: Basic initialization")
    config = AdaptiveQGFLConfig(task='binary', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)
    callback = AdaptiveTrainingCallback(loss_fn, config, use_wandb=False)

    assert callback.epoch == 0
    assert callback.loss_fn is loss_fn
    print("  Callback initialized: ✓")
    print("  PASSED")

    # Test 2: Mock trainer with metrics
    print("\nTest 2: _extract_metrics from mock trainer")

    class MockBox:
        def __init__(self):
            self.r = [0.95, 0.60]  # Per-class recall
            self.p = [0.80, 0.75]  # Per-class precision

    class MockMetrics:
        def __init__(self):
            self.box = MockBox()

    class MockTrainer:
        def __init__(self):
            self.epoch = 10
            self.epochs = 200
            self.metrics = MockMetrics()

    trainer = MockTrainer()
    metrics = callback._extract_metrics(trainer)

    assert metrics is not None, "Should extract metrics from mock trainer"
    assert 0 in metrics['recall'], "Should have recall for class 0"
    assert 1 in metrics['recall'], "Should have recall for class 1"
    assert abs(metrics['recall'][0] - 0.95) < 0.01
    assert abs(metrics['recall'][1] - 0.60) < 0.01
    print(f"  Extracted recall: {metrics['recall']}")
    print(f"  Extracted precision: {metrics['precision']}")
    print(f"  Computed F1: {metrics['f1']}")
    print("  PASSED")

    # Test 3: Full on_fit_epoch_end simulation
    print("\nTest 3: on_fit_epoch_end simulation")

    config = AdaptiveQGFLConfig(
        task='binary',
        warmup_epochs=0,
        freeze_final_epochs=0,
        adaptation_momentum=0.0,
        debug=False
    )
    loss_fn = DynamicAdaptiveYOLOLoss(config)
    callback = AdaptiveTrainingCallback(loss_fn, config, use_wandb=False)

    initial_alpha = loss_fn.class_alphas[1].item()

    # Simulate low recall on infected class
    class MockBox2:
        def __init__(self):
            self.r = [0.95, 0.40]
            self.p = [0.80, 0.70]

    class MockMetrics2:
        def __init__(self):
            self.box = MockBox2()

    trainer.metrics = MockMetrics2()

    info = callback.on_fit_epoch_end(trainer)

    new_alpha = loss_fn.class_alphas[1].item()
    assert info is not None, "Should return adaptation info"
    assert info.get('adapted') == True, "Should have adapted"
    assert new_alpha > initial_alpha, "Alpha should increase with low recall"
    print(f"  Adaptation triggered: {info.get('adapted')}")
    print(f"  Alpha changed: {initial_alpha:.3f} -> {new_alpha:.3f}")
    print("  PASSED")

    # Test 4: get_callback_dict
    print("\nTest 4: get_callback_dict")
    callbacks = callback.get_callback_dict()

    assert 'on_train_start' in callbacks
    assert 'on_fit_epoch_end' in callbacks
    assert callable(callbacks['on_train_start'])
    assert callable(callbacks['on_fit_epoch_end'])
    print(f"  Callbacks: {list(callbacks.keys())}")
    print("  PASSED")

    # Test 5: Factory function
    print("\nTest 5: create_adaptive_callback factory")
    config = AdaptiveQGFLConfig(task='staging', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    callback = create_adaptive_callback(loss_fn, config, use_wandb=False)
    assert isinstance(callback, AdaptiveTrainingCallback)
    print("  Factory creates callback: ✓")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL CALLBACK TESTS PASSED")
    print("=" * 60)
