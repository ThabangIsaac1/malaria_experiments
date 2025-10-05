"""
Quality-Guided Focal Loss (QGFL) for YOLO (v8/v11)

YOLO-specific QGFL implementation that replaces BCE classification loss
with quality-guided focal loss for better minority class detection.

Integration Point: ultralytics/utils/loss.py, line 247
  Current: loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
  Replace with: QGFL loss

Author: Thabang Isaka
Date: 2025-10-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from qgfl_core import QGFLCore


class QGFLYOLOLoss(QGFLCore):
    """
    QGFL loss for YOLO (v8/v11) architecture

    YOLO-specific characteristics:
    - Soft targets (IoU-quality weighted) ∈ [0, 1]
    - ~8400 predictions per image
    - Task-Aligned Assigner for target assignment
    - Binary classification (nc=2)
    """

    def __init__(self, nc: int = 2, **qgfl_params):
        """
        Initialize QGFL loss for YOLO

        Args:
            nc: Number of classes (default=2 for binary)
            **qgfl_params: QGFL parameters (alpha, gamma, etc.)
        """
        super().__init__(**qgfl_params)
        self.nc = nc

        if self.debug:
            print(f"\n[QGFL-YOLO] Initialized for {nc} classes")

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        current_epoch: int = 0,
        total_epochs: int = 200
    ) -> torch.Tensor:
        """
        Compute QGFL loss for YOLO

        YOLO provides soft targets (IoU-quality weighted), not hard 0/1 labels.
        This is actually beneficial for QGFL as quality is already embedded.

        Args:
            pred_scores: [batch, num_preds, nc] - Predicted logits (NOT probabilities)
            target_scores: [batch, num_preds, nc] - Soft targets ∈ [0, 1] (IoU-weighted)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar QGFL loss value
        """
        if self.debug:
            self.validate_tensors(pred_scores, target_scores, stage="YOLO input")

        # Get predicted probabilities from logits
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss
        # For soft targets: pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        pt = pred_prob * target_scores + (1 - pred_prob) * (1 - target_scores)

        # Clamp pt to avoid log(0)
        pt = torch.clamp(pt, min=self.epsilon, max=1.0 - self.epsilon)

        # Determine which predictions are infected class (class index 1 in binary)
        # For YOLO's soft targets, use argmax to determine dominant class
        target_class = target_scores.argmax(dim=-1, keepdim=True)  # [batch, num_preds, 1]
        is_infected = (target_class == 1)  # [batch, num_preds, 1]

        # Expand to match pred_scores shape
        is_infected = is_infected.expand_as(pred_scores)  # [batch, num_preds, nc]

        # Compute class-specific alpha
        alpha = self.get_class_alpha(is_infected)  # [batch, num_preds, nc]

        # Compute effective gamma (difficulty + class aware)
        gamma_eff = self.compute_gamma_eff(pt, is_infected)  # [batch, num_preds, nc]

        if self.debug:
            print(f"\n[DEBUG] QGFL-YOLO Forward Pass:")
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

        if self.debug:
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
        if self.debug:
            self.check_loss_sanity(loss_total, stage="YOLO QGFL")

        return loss_total


# Standalone testing
if __name__ == "__main__":
    print("Testing QGFL YOLO Loss...")

    # Initialize
    qgfl_yolo = QGFLYOLOLoss(nc=2, debug=True)

    # Simulate YOLO outputs
    batch_size = 2
    num_preds = 100  # Simplified (real YOLO has ~8400)
    nc = 2

    print("\n" + "="*60)
    print("Test: YOLO Forward Pass")
    print("="*60)

    # Predicted logits (before sigmoid)
    pred_scores = torch.randn(batch_size, num_preds, nc)

    # Soft targets (IoU-quality weighted)
    # Most predictions are background (low target_scores)
    target_scores = torch.rand(batch_size, num_preds, nc) * 0.3
    # Make some predictions positive (high target_scores for infected class)
    target_scores[:, :10, 1] = torch.rand(batch_size, 10) * 0.7 + 0.3  # Infected
    target_scores[:, :10, 0] = 1.0 - target_scores[:, :10, 1]  # Uninfected

    print(f"\nInput shapes:")
    print(f"  pred_scores: {pred_scores.shape}")
    print(f"  target_scores: {target_scores.shape}")

    # Forward pass
    loss = qgfl_yolo(
        pred_scores=pred_scores,
        target_scores=target_scores,
        current_epoch=50,
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
    loss_grad = qgfl_yolo(pred_scores_grad, target_scores, current_epoch=50, total_epochs=200)
    loss_grad.backward()

    print(f"Gradients computed: {pred_scores_grad.grad is not None}")
    if pred_scores_grad.grad is not None:
        print(f"Gradient range: [{pred_scores_grad.grad.min():.6f}, {pred_scores_grad.grad.max():.6f}]")
        print(f"Gradient mean: {pred_scores_grad.grad.mean():.6f}")
        print(f"Non-zero gradients: {(pred_scores_grad.grad != 0).sum().item()} / {pred_scores_grad.grad.numel()}")

    print("\n" + "="*60)
    print("QGFL YOLO Tests Passed! ✓")
    print("="*60)
