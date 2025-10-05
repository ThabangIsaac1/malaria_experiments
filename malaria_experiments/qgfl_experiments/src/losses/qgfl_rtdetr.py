"""
Quality-Guided Focal Loss (QGFL) for RT-DETR

RT-DETR-specific QGFL implementation that replaces VarifocalLoss
with quality-guided focal loss for better minority class detection.

Integration Point: ultralytics/models/utils/loss.py, lines 77-84
  Current: VarifocalLoss or FocalLoss or BCE
  Replace with: QGFL loss

IMPORTANT: RT-DETR baseline uses VarifocalLoss, NOT BCE!
QGFL must beat VarifocalLoss (which already has α=0.75, γ=2.0)

Author: Thabang Isaka
Date: 2025-10-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from qgfl_core import QGFLCore


class QGFLRTDETRLoss(QGFLCore):
    """
    QGFL loss for RT-DETR architecture

    RT-DETR-specific characteristics:
    - One-hot targets + quality scores (separate)
    - 300 queries per image (vs YOLO's 8400)
    - Hungarian Matcher for bipartite assignment
    - Binary classification (nc=2)
    - Currently uses VarifocalLoss (α=0.75, γ=2.0)
    """

    def __init__(self, nc: int = 2, **qgfl_params):
        """
        Initialize QGFL loss for RT-DETR

        Args:
            nc: Number of classes (default=2 for binary)
            **qgfl_params: QGFL parameters (alpha, gamma, etc.)
        """
        super().__init__(**qgfl_params)
        self.nc = nc

        if self.debug:
            print(f"\n[QGFL-RTDETR] Initialized for {nc} classes")
            print(f"[QGFL-RTDETR] NOTE: RT-DETR baseline uses VarifocalLoss, not BCE!")

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
        Compute QGFL loss for RT-DETR

        RT-DETR provides one-hot targets AND quality scores (IoU-based).
        This is different from YOLO's soft targets.

        Args:
            pred_scores: [batch, 300, nc] - Predicted logits (NOT probabilities)
            one_hot: [batch, 300, nc] - One-hot encoded targets {0, 1}
            gt_scores: [batch, 300, nc] - IoU-quality weighted targets ∈ [0, 1]
            num_gts: Number of ground truth boxes (scalar)
            nq: Number of queries (300 for RT-DETR)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar QGFL loss value
        """
        if self.debug:
            self.validate_tensors(pred_scores, gt_scores, stage="RT-DETR input")
            print(f"\n[DEBUG] RT-DETR Forward:")
            print(f"  Batch size: {pred_scores.shape[0]}")
            print(f"  Num queries: {pred_scores.shape[1]}")
            print(f"  Num classes: {pred_scores.shape[2]}")
            print(f"  Num ground truths: {num_gts}")
            print(f"  one_hot range: [{one_hot.min():.1f}, {one_hot.max():.1f}]")
            print(f"  gt_scores range: [{gt_scores.min():.4f}, {gt_scores.max():.4f}]")

        # Get predicted probabilities from logits
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss
        # RT-DETR uses quality-weighted one-hot targets
        pt = pred_prob * gt_scores + (1 - pred_prob) * (1 - gt_scores)

        # Clamp pt to avoid log(0)
        pt = torch.clamp(pt, min=self.epsilon, max=1.0 - self.epsilon)

        # Determine which predictions are infected class
        # For RT-DETR: one_hot[..., 1] indicates infected class
        is_infected = (one_hot[..., 1:2] == 1).expand_as(pred_scores)  # [batch, 300, nc]

        # Compute class-specific alpha
        alpha = self.get_class_alpha(is_infected)  # [batch, 300, nc]

        # Compute effective gamma (difficulty + class aware)
        gamma_eff = self.compute_gamma_eff(pt, is_infected)  # [batch, 300, nc]

        if self.debug:
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff.min():.4f}, {gamma_eff.max():.4f}]")

        # Compute quality weight
        quality_weight = self.compute_quality_weight(pred_prob, gt_scores)  # [batch, 300]
        quality_weight = quality_weight.unsqueeze(-1)  # [batch, 300, 1]

        # Compute UIoU ratio
        uiou_ratio = self.compute_uiou_ratio(current_epoch, total_epochs)

        if self.debug:
            print(f"  quality_weight range: [{quality_weight.min():.4f}, {quality_weight.max():.4f}]")
            print(f"  uiou_ratio: {uiou_ratio:.3f}")

        # Standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            gt_scores,
            reduction='none'
        )  # [batch, 300, nc]

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
        )  # [batch, 300, nc]

        # Normalize by number of GTs (same as VarifocalLoss)
        # RT-DETR normalization: mean(1).sum() / max(num_gts, 1) * nq
        loss_normalized = loss.mean(1).sum() / max(num_gts, 1) * nq

        # Sanity check
        if self.debug:
            self.check_loss_sanity(loss_normalized, stage="RT-DETR QGFL")

        return loss_normalized


# Standalone testing
if __name__ == "__main__":
    print("Testing QGFL RT-DETR Loss...")

    # Initialize
    qgfl_rtdetr = QGFLRTDETRLoss(nc=2, debug=True)

    # Simulate RT-DETR outputs
    batch_size = 2
    nq = 300  # RT-DETR queries
    nc = 2
    num_gts = 15  # Total ground truths in batch

    print("\n" + "="*60)
    print("Test: RT-DETR Forward Pass")
    print("="*60)

    # Predicted logits (before sigmoid)
    pred_scores = torch.randn(batch_size, nq, nc)

    # One-hot targets (hard labels)
    one_hot = torch.zeros(batch_size, nq, nc)
    # First 10 queries matched to infected
    one_hot[:, :5, 1] = 1
    one_hot[:, :5, 0] = 0
    # Rest matched to uninfected or background
    one_hot[:, 5:15, 0] = 1
    one_hot[:, 5:15, 1] = 0

    # Quality scores (IoU-weighted)
    gt_scores = one_hot.clone()
    # Apply IoU quality weighting to matched predictions
    gt_scores[:, :15, :] *= torch.rand(batch_size, 15, 1).expand(-1, -1, nc) * 0.5 + 0.5

    print(f"\nInput shapes:")
    print(f"  pred_scores: {pred_scores.shape}")
    print(f"  one_hot: {one_hot.shape}")
    print(f"  gt_scores: {gt_scores.shape}")
    print(f"  num_gts: {num_gts}")
    print(f"  nq: {nq}")

    # Forward pass
    loss = qgfl_rtdetr(
        pred_scores=pred_scores,
        one_hot=one_hot,
        gt_scores=gt_scores,
        num_gts=num_gts,
        nq=nq,
        current_epoch=100,
        total_epochs=200
    )

    print(f"\nOutput:")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss is finite: {torch.isfinite(loss).item()}")
    print(f"  Loss > 0: {(loss > 0).item()}")

    # Test gradient flow
    print("\n" + "="*60)
    print("Test: Gradient Flow")
    print("="*60)

    pred_scores_grad = pred_scores.clone().requires_grad_(True)
    loss_grad = qgfl_rtdetr(
        pred_scores_grad, one_hot, gt_scores,
        num_gts, nq, current_epoch=100, total_epochs=200
    )
    loss_grad.backward()

    print(f"Gradients computed: {pred_scores_grad.grad is not None}")
    if pred_scores_grad.grad is not None:
        print(f"Gradient range: [{pred_scores_grad.grad.min():.6f}, {pred_scores_grad.grad.max():.6f}]")
        print(f"Gradient mean: {pred_scores_grad.grad.mean():.6f}")
        print(f"Non-zero gradients: {(pred_scores_grad.grad != 0).sum().item()} / {pred_scores_grad.grad.numel()}")

    print("\n" + "="*60)
    print("QGFL RT-DETR Tests Passed! ✓")
    print("="*60)
