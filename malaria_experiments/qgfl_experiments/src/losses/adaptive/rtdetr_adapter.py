"""
Dynamic Adaptive QGFL for RT-DETR

RT-DETR-specific adapter that extends DynamicAdaptiveQGFLCore with:
- One-hot target + quality score handling
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


class DynamicAdaptiveRTDETRLoss(DynamicAdaptiveQGFLCore):
    """
    Dynamic Adaptive QGFL for RT-DETR architecture

    RT-DETR-specific characteristics:
    - One-hot targets + quality scores (separate tensors)
    - 300 queries per image (vs YOLO's 8400)
    - Hungarian Matcher for bipartite assignment
    - RT-DETR baseline uses VarifocalLoss (alpha=0.75, gamma=2.0)

    Key differences from static QGFLRTDETRLoss:
    - Uses per-class alpha/gamma that adapt during training
    - Adaptation triggered via adapt_parameters() after validation
    """

    def __init__(self, config: AdaptiveQGFLConfig):
        """
        Initialize Dynamic Adaptive RT-DETR Loss

        Args:
            config: AdaptiveQGFLConfig with task/dataset/model settings
        """
        super().__init__(config)

        if config.debug:
            print(f"\n[DynamicAdaptiveRTDETR] Initialized for {config.nc} classes")
            print(f"  Model: {config.model_type}")
            print(f"  Task: {config.task}")
            print(f"  NOTE: RT-DETR baseline uses VarifocalLoss, not BCE!")

    def forward(
        self,
        pred_scores: torch.Tensor,
        one_hot: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        nq: int,
        current_epoch: int = 0,
        total_epochs: int = 200
    ) -> torch.Tensor:
        """
        Compute Dynamic Adaptive QGFL loss for RT-DETR

        Uses per-class alpha and gamma values that are adapted during training
        based on validation metrics.

        Args:
            pred_scores: [batch, 300, nc] - Predicted logits (NOT probabilities)
            one_hot: [batch, 300, nc] - One-hot encoded targets {0, 1}
            gt_scores: [batch, 300, nc] - IoU-quality weighted targets in [0, 1]
            num_gts: Number of ground truth boxes (scalar)
            nq: Number of queries (300 for RT-DETR)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar QGFL loss value
        """
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs

        if self.config.debug:
            self.validate_tensors(pred_scores, gt_scores, stage="AdaptiveRTDETR input")
            print(f"\n[DEBUG] AdaptiveRTDETR Forward:")
            print(f"  Batch size: {pred_scores.shape[0]}")
            print(f"  Num queries: {pred_scores.shape[1]}")
            print(f"  Num classes: {pred_scores.shape[2]}")
            print(f"  Num ground truths: {num_gts}")

        # Get predicted probabilities from logits
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss
        # RT-DETR uses quality-weighted one-hot targets
        pt = pred_prob * gt_scores + (1 - pred_prob) * (1 - gt_scores)
        pt = torch.clamp(pt, min=self.epsilon, max=1.0 - self.epsilon)

        # Determine class for each query from one-hot encoding
        # one_hot shape: [batch, nq, nc]
        target_class = one_hot.argmax(dim=-1)  # [batch, nq]

        # Get per-class alpha and gamma tensors
        alpha_per_sample = self._get_alpha_for_classes(target_class, pred_scores.device)
        gamma_per_sample = self._get_gamma_for_classes(target_class, pred_scores.device)

        # Expand to match pred_scores shape [batch, nq, nc]
        alpha = alpha_per_sample.unsqueeze(-1).expand_as(pred_scores)
        gamma_base = gamma_per_sample.unsqueeze(-1).expand_as(pred_scores)

        # Apply difficulty-aware gamma adjustment
        difficulty = 1.0 - pt
        difficulty_adjusted = torch.clamp(
            difficulty - self.current_threshold,
            min=0.0
        )
        if self.current_threshold < 1.0:
            difficulty_norm = difficulty_adjusted / (1.0 - self.current_threshold)
        else:
            difficulty_norm = difficulty_adjusted

        gamma_eff = 2.0 + (gamma_base - 2.0) * difficulty_norm

        if self.config.debug:
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff.min():.4f}, {gamma_eff.max():.4f}]")

        # Compute quality weight
        quality_weight = self.compute_quality_weight(pred_prob, gt_scores)  # [batch, nq]
        quality_weight = quality_weight.unsqueeze(-1)  # [batch, nq, 1]

        # Compute UIoU ratio
        uiou_ratio = self.compute_uiou_ratio(current_epoch, total_epochs)

        if self.config.debug:
            print(f"  quality_weight range: [{quality_weight.min():.4f}, {quality_weight.max():.4f}]")
            print(f"  uiou_ratio: {uiou_ratio:.3f}")

        # Standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            gt_scores,
            reduction='none'
        )  # [batch, nq, nc]

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
        )  # [batch, nq, nc]

        # Normalize by number of GTs (same as VarifocalLoss)
        # RT-DETR normalization: mean(1).sum() / max(num_gts, 1) * nq
        loss_normalized = loss.mean(1).sum() / max(num_gts, 1) * nq

        # Sanity check
        if self.config.debug:
            self.check_loss_sanity(loss_normalized, stage="AdaptiveRTDETR QGFL")

        return loss_normalized

    def _get_alpha_for_classes(
        self,
        class_ids: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Get alpha values for each query based on its class

        Args:
            class_ids: [batch, nq] - Class IDs for each query
            device: Target device

        Returns:
            alpha: [batch, nq] - Alpha values
        """
        batch, nq = class_ids.shape
        alpha = torch.zeros(batch, nq, dtype=torch.float32, device=device)

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
        Get gamma values for each query based on its class

        Args:
            class_ids: [batch, nq] - Class IDs for each query
            device: Target device

        Returns:
            gamma: [batch, nq] - Gamma values
        """
        batch, nq = class_ids.shape
        gamma = torch.zeros(batch, nq, dtype=torch.float32, device=device)

        for c in range(self.config.nc):
            mask = (class_ids == c)
            gamma[mask] = self.class_gammas[c].item()

        return gamma


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DynamicAdaptiveRTDETRLoss")
    print("=" * 60)

    # Test 1: Basic forward pass (binary)
    print("\nTest 1: Basic forward pass (binary)")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1', model_type='rtdetr', debug=False)
    loss_fn = DynamicAdaptiveRTDETRLoss(config)

    batch_size = 2
    nq = 300
    nc = 2
    num_gts = 15

    pred_scores = torch.randn(batch_size, nq, nc)

    # One-hot targets
    one_hot = torch.zeros(batch_size, nq, nc)
    one_hot[:, :5, 1] = 1  # First 5: infected
    one_hot[:, :5, 0] = 0
    one_hot[:, 5:15, 0] = 1  # Next 10: uninfected
    one_hot[:, 5:15, 1] = 0

    # Quality scores (IoU-weighted)
    gt_scores = one_hot.clone()
    gt_scores[:, :15, :] *= torch.rand(batch_size, 15, 1).expand(-1, -1, nc) * 0.5 + 0.5

    loss = loss_fn(pred_scores, one_hot, gt_scores, num_gts, nq, current_epoch=50, total_epochs=200)

    assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    print(f"  Loss: {loss.item():.6f} ✓")
    print("  PASSED")

    # Test 2: Gradient flow
    print("\nTest 2: Gradient flow")
    pred_grad = pred_scores.clone().requires_grad_(True)
    loss_grad = loss_fn(pred_grad, one_hot, gt_scores, num_gts, nq, current_epoch=50, total_epochs=200)
    loss_grad.backward()

    assert pred_grad.grad is not None, "Gradients should be computed"
    assert torch.isfinite(pred_grad.grad).all(), "Gradients should be finite"
    print(f"  Gradient range: [{pred_grad.grad.min():.6f}, {pred_grad.grad.max():.6f}]")
    print("  PASSED")

    # Test 3: Parameters change after adaptation
    print("\nTest 3: Parameters change after adaptation")
    config = AdaptiveQGFLConfig(
        task='binary',
        model_type='rtdetr',
        warmup_epochs=0,
        freeze_final_epochs=0,
        adaptation_momentum=0.0,
        debug=False
    )
    loss_fn = DynamicAdaptiveRTDETRLoss(config)

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
    config = AdaptiveQGFLConfig(task='staging', dataset='d1', model_type='rtdetr', debug=False)
    loss_fn = DynamicAdaptiveRTDETRLoss(config)

    nc = 5
    pred_scores = torch.randn(batch_size, nq, nc)
    one_hot = torch.zeros(batch_size, nq, nc)
    # Distribute some samples across classes
    for c in range(nc):
        one_hot[:, c*3:(c+1)*3, c] = 1
    gt_scores = one_hot.clone() * (torch.rand(batch_size, nq, 1) * 0.5 + 0.5)

    loss = loss_fn(pred_scores, one_hot, gt_scores, num_gts=15, nq=nq, current_epoch=50, total_epochs=200)

    assert torch.isfinite(loss), f"Loss should be finite for multiclass"
    assert loss > 0, f"Loss should be positive for multiclass"
    print(f"  Loss (5 classes): {loss.item():.6f} ✓")
    print("  PASSED")

    # Test 5: Normalization matches RT-DETR VarifocalLoss pattern
    print("\nTest 5: Normalization by num_gts")
    config = AdaptiveQGFLConfig(task='binary', model_type='rtdetr', debug=False)
    loss_fn = DynamicAdaptiveRTDETRLoss(config)

    # Same inputs, different num_gts
    loss_5gts = loss_fn(pred_scores[:, :, :2], one_hot[:, :, :2], gt_scores[:, :, :2],
                        num_gts=5, nq=300, current_epoch=50, total_epochs=200)
    loss_20gts = loss_fn(pred_scores[:, :, :2], one_hot[:, :, :2], gt_scores[:, :, :2],
                         num_gts=20, nq=300, current_epoch=50, total_epochs=200)

    # With more GTs, normalized loss should be different
    assert loss_5gts != loss_20gts, "Normalization by num_gts should affect loss"
    print(f"  Loss with 5 GTs: {loss_5gts.item():.6f}")
    print(f"  Loss with 20 GTs: {loss_20gts.item():.6f}")
    print("  PASSED")

    # Test 6: Verify per-class alpha used correctly
    print("\nTest 6: Verify per-class alpha usage")
    config = AdaptiveQGFLConfig(task='binary', model_type='rtdetr', debug=False)
    loss_fn = DynamicAdaptiveRTDETRLoss(config)

    # Set distinct alpha values
    with torch.no_grad():
        loss_fn.class_alphas[0].fill_(0.2)  # Majority
        loss_fn.class_alphas[1].fill_(0.8)  # Minority

    class_ids = torch.tensor([[0, 1, 0, 1, 1]])  # 1 batch, 5 queries
    alphas = loss_fn._get_alpha_for_classes(class_ids, torch.device('cpu'))

    expected = torch.tensor([[0.2, 0.8, 0.2, 0.8, 0.8]])
    assert torch.allclose(alphas, expected), f"Expected {expected}, got {alphas}"
    print(f"  Class IDs: {class_ids.tolist()}")
    print(f"  Alphas: {alphas.tolist()}")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL RT-DETR ADAPTER TESTS PASSED")
    print("=" * 60)
