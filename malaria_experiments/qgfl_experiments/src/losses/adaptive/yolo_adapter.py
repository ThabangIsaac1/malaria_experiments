"""
Dynamic Adaptive QGFL for YOLO (v8/v11)

YOLO-specific adapter that extends DynamicAdaptiveQGFLCore with:
- Soft target handling (IoU-quality weighted targets)
- Per-class alpha/gamma from adaptive core
- Compatible with existing monkey-patching integration

Author: Thabang Isaka
Date: 2025-11-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.losses.adaptive.core import DynamicAdaptiveQGFLCore
from src.losses.adaptive.config import AdaptiveQGFLConfig


class DynamicAdaptiveYOLOLoss(DynamicAdaptiveQGFLCore):
    """
    Dynamic Adaptive QGFL for YOLO architecture

    YOLO-specific characteristics:
    - Soft targets (IoU-quality weighted) in [0, 1]
    - ~8400 predictions per image
    - Task-Aligned Assigner for target assignment
    - Supports nc=2 (binary) and nc>2 (multiclass)

    Key differences from static QGFLYOLOLoss:
    - Uses per-class alpha/gamma that adapt during training
    - Adaptation triggered via adapt_parameters() after validation
    """

    def __init__(self, config: AdaptiveQGFLConfig):
        """
        Initialize Dynamic Adaptive YOLO Loss

        Args:
            config: AdaptiveQGFLConfig with task/dataset/model settings
        """
        super().__init__(config)

        if config.debug:
            print(f"\n[DynamicAdaptiveYOLO] Initialized for {config.nc} classes")
            print(f"  Model: {config.model_type}")
            print(f"  Task: {config.task}")

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        current_epoch: int = 0,
        total_epochs: int = 200
    ) -> torch.Tensor:
        """
        Compute Dynamic Adaptive QGFL loss for YOLO

        Uses per-class alpha and gamma values that are adapted during training
        based on validation metrics.

        Args:
            pred_scores: [batch, num_preds, nc] - Predicted logits (NOT probabilities)
            target_scores: [batch, num_preds, nc] - Soft targets in [0, 1] (IoU-weighted)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar QGFL loss value
        """
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        if self.config.debug:
            self.validate_tensors(pred_scores, target_scores, stage="AdaptiveYOLO input")

        # Get predicted probabilities from logits
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss
        # For soft targets: pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        pt = pred_prob * target_scores + (1 - pred_prob) * (1 - target_scores)
        pt = torch.clamp(pt, min=self.epsilon, max=1.0 - self.epsilon)

        # Determine class for each prediction using argmax
        target_class = target_scores.argmax(dim=-1)  # [batch, num_preds]

        # Get per-class alpha and gamma tensors
        # Shape: [batch, num_preds]
        alpha_per_sample = self._get_alpha_for_classes(target_class, pred_scores.device)
        gamma_per_sample = self._get_gamma_for_classes(target_class, pred_scores.device)

        # Expand to match pred_scores shape [batch, num_preds, nc]
        alpha = alpha_per_sample.unsqueeze(-1).expand_as(pred_scores)
        gamma_base = gamma_per_sample.unsqueeze(-1).expand_as(pred_scores)

        # Apply difficulty-aware gamma adjustment
        # gamma_eff = gamma_base + difficulty_boost for hard examples
        difficulty = 1.0 - pt
        difficulty_adjusted = torch.clamp(
            difficulty - self.current_threshold,
            min=0.0
        )
        if self.current_threshold < 1.0:
            difficulty_norm = difficulty_adjusted / (1.0 - self.current_threshold)
        else:
            difficulty_norm = difficulty_adjusted

        # Interpolate: for very easy samples (pt near 1), use base gamma
        # For hard samples (low pt), use full class gamma
        gamma_eff = 2.0 + (gamma_base - 2.0) * difficulty_norm

        if self.config.debug:
            print(f"\n[DEBUG] AdaptiveYOLO Forward Pass:")
            print(f"  Batch size: {pred_scores.shape[0]}")
            print(f"  Num predictions: {pred_scores.shape[1]}")
            print(f"  Num classes: {pred_scores.shape[2]}")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff.min():.4f}, {gamma_eff.max():.4f}]")

        # Compute quality weight
        quality_weight = self.compute_quality_weight(pred_prob, target_scores)  # [batch, num_preds]
        quality_weight = quality_weight.unsqueeze(-1)  # [batch, num_preds, 1]

        # Compute UIoU ratio
        uiou_ratio = self.compute_uiou_ratio(current_epoch, total_epochs)

        if self.config.debug:
            print(f"  quality_weight range: [{quality_weight.min():.4f}, {quality_weight.max():.4f}]")
            print(f"  uiou_ratio: {uiou_ratio:.3f}")

        # Standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            target_scores,
            reduction='none'
        )  # [batch, num_preds, nc]

        # Modulating factor: (1 - pt)^gamma_eff
        modulating_factor = (1.0 - pt) ** gamma_eff

        # Apply all QGFL components:
        # QGFL = alpha * (1-pt)^gamma_eff * (1 + quality_weight) * uiou_ratio * BCE
        loss = (
            alpha *
            modulating_factor *
            (1.0 + quality_weight) *
            uiou_ratio *
            bce_loss
        )  # [batch, num_preds, nc]

        # Sum over all elements
        loss_total = loss.sum()

        # Sanity check
        if self.config.debug:
            self.check_loss_sanity(loss_total, stage="AdaptiveYOLO QGFL")

        return loss_total

    def _get_alpha_for_classes(
        self,
        class_ids: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get alpha values for each sample based on its class

        Args:
            class_ids: [batch, num_preds] - Class IDs for each prediction
            device: Target device

        Returns:
            alpha: [batch, num_preds] - Alpha values
        """
        batch, num_preds = class_ids.shape
        alpha = torch.zeros(batch, num_preds, dtype=torch.float32, device=device)

        for c in range(self.config.nc):
            mask = (class_ids == c)
            alpha[mask] = self.class_alphas[c].item()

        return alpha

    def _get_gamma_for_classes(
        self,
        class_ids: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get gamma values for each sample based on its class

        Args:
            class_ids: [batch, num_preds] - Class IDs for each prediction
            device: Target device

        Returns:
            gamma: [batch, num_preds] - Gamma values
        """
        batch, num_preds = class_ids.shape
        gamma = torch.zeros(batch, num_preds, dtype=torch.float32, device=device)

        for c in range(self.config.nc):
            mask = (class_ids == c)
            gamma[mask] = self.class_gammas[c].item()

        return gamma


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DynamicAdaptiveYOLOLoss")
    print("=" * 60)

    # Test 1: Basic forward pass (binary)
    print("\nTest 1: Basic forward pass (binary)")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    batch_size = 2
    num_preds = 100
    nc = 2

    pred_scores = torch.randn(batch_size, num_preds, nc)
    target_scores = torch.rand(batch_size, num_preds, nc) * 0.3
    # Some infected samples
    target_scores[:, :10, 1] = torch.rand(batch_size, 10) * 0.7 + 0.3
    target_scores[:, :10, 0] = 1.0 - target_scores[:, :10, 1]

    loss = loss_fn(pred_scores, target_scores, current_epoch=50, total_epochs=200)

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    print(f"  Loss: {loss.item():.6f} ✓")
    print("  PASSED")

    # Test 2: Gradient flow
    print("\nTest 2: Gradient flow")
    pred_grad = pred_scores.clone().requires_grad_(True)
    loss_grad = loss_fn(pred_grad, target_scores, current_epoch=50, total_epochs=200)
    loss_grad.backward()

    assert pred_grad.grad is not None, "Gradients should be computed"
    assert torch.isfinite(pred_grad.grad).all(), "Gradients should be finite"
    print(f"  Gradient range: [{pred_grad.grad.min():.6f}, {pred_grad.grad.max():.6f}]")
    print("  PASSED")

    # Test 3: Different alpha/gamma after adaptation
    print("\nTest 3: Parameters change after adaptation")
    config = AdaptiveQGFLConfig(
        task='binary',
        warmup_epochs=0,
        freeze_final_epochs=0,
        adaptation_momentum=0.0,
        debug=False
    )
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    initial_alpha = loss_fn.class_alphas[1].item()

    # Run one adaptation with low recall
    metrics = {
        'recall': {0: 0.95, 1: 0.40},
        'precision': {0: 0.80, 1: 0.70},
        'f1': {0: 0.87, 1: 0.52}
    }
    loss_fn.adapt_parameters(epoch=10, metrics=metrics, total_epochs=200)

    new_alpha = loss_fn.class_alphas[1].item()
    assert new_alpha != initial_alpha, f"Alpha should change after adaptation"
    print(f"  Alpha before: {initial_alpha:.3f}")
    print(f"  Alpha after: {new_alpha:.3f}")
    print("  PASSED")

    # Test 4: Multiclass (staging)
    print("\nTest 4: Multiclass forward pass (staging)")
    config = AdaptiveQGFLConfig(task='staging', dataset='d1', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    nc = 5
    pred_scores = torch.randn(batch_size, num_preds, nc)
    target_scores = torch.rand(batch_size, num_preds, nc)
    target_scores = F.softmax(target_scores, dim=-1)  # Normalize

    loss = loss_fn(pred_scores, target_scores, current_epoch=50, total_epochs=200)

    assert torch.isfinite(loss), f"Loss should be finite for multiclass"
    assert loss > 0, f"Loss should be positive for multiclass"
    print(f"  Loss (5 classes): {loss.item():.6f} ✓")
    print("  PASSED")

    # Test 5: Verify per-class alpha used correctly
    print("\nTest 5: Verify per-class alpha usage")
    config = AdaptiveQGFLConfig(task='binary', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    # Set distinct alpha values
    with torch.no_grad():
        loss_fn.class_alphas[0].fill_(0.2)  # Majority
        loss_fn.class_alphas[1].fill_(0.8)  # Minority

    class_ids = torch.tensor([[0, 1, 0, 1, 1]])  # 1 batch, 5 preds
    alphas = loss_fn._get_alpha_for_classes(class_ids, torch.device('cpu'))

    expected = torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.8]])
    assert torch.allclose(alphas, expected), f"Expected {expected}, got {alphas}"
    print(f"  Class IDs: {class_ids.tolist()}")
    print(f"  Alphas: {alphas.tolist()}")
    print("  PASSED")

    # Test 6: UIoU decay during forward
    print("\nTest 6: UIoU decay in forward pass")
    config = AdaptiveQGFLConfig(task='binary', debug=False)
    loss_fn = DynamicAdaptiveYOLOLoss(config)

    pred_scores = torch.randn(2, 50, 2)
    target_scores = torch.rand(2, 50, 2)

    # Early epoch
    uiou_early = loss_fn.compute_uiou_ratio(0, 200)
    # Late epoch
    uiou_late = loss_fn.compute_uiou_ratio(199, 200)

    assert uiou_early > uiou_late, f"UIoU should decay: {uiou_early} -> {uiou_late}"
    print(f"  UIoU at epoch 0: {uiou_early:.3f}")
    print(f"  UIoU at epoch 199: {uiou_late:.3f}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL YOLO ADAPTER TESTS PASSED")
    print("=" * 60)
