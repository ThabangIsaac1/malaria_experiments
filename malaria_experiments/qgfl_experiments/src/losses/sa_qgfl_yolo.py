"""
Species-Adaptive Quality-Guided Focal Loss (SA-QGFL) for YOLO (v8/v11)

Multi-class YOLO-specific SA-QGFL implementation for species-level Plasmodium detection.
Replaces BCE classification loss with species-adaptive focal loss.

Key differences from binary QGFL:
1. Per-class alpha (PWA) instead of binary infected/uninfected
2. Per-class gamma (MCP) based on species morphological complexity
3. Hierarchical Quality Scoring (HQS) for two-tier quality assessment

Integration Point: ultralytics/utils/loss.py, line 247
  Current: loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
  Replace with: SA-QGFL loss

Author: Thabang Isaka
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import sys
from pathlib import Path

# Handle both module import and direct execution
try:
    from .sa_qgfl_core import SAQGFLCore
    from .sa_qgfl_config import NC, CLASS_NAMES
except ImportError:
    # Direct execution - add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from sa_qgfl_core import SAQGFLCore
    from sa_qgfl_config import NC, CLASS_NAMES


class SAQGFLYOLOLoss(SAQGFLCore):
    """
    SA-QGFL loss for YOLO (v8/v11) architecture - Multi-class version

    YOLO-specific characteristics:
    - Soft targets (IoU-quality weighted) ∈ [0, 1]
    - ~8400 predictions per image
    - Task-Aligned Assigner for target assignment
    - Multi-class (nc=5 for D3 species detection)

    Novel components:
    - MCP: Per-species gamma based on morphological detection difficulty
    - PWA: Per-species alpha based on prevalence + clinical importance
    - HQS: Hierarchical quality scoring (infection → species)
    """

    def __init__(self, nc: int = NC, **saqgfl_params):
        """
        Initialize SA-QGFL loss for YOLO

        Args:
            nc: Number of classes (default=5 for D3 species)
            **saqgfl_params: SA-QGFL parameters (class_alphas, class_gammas, etc.)
        """
        # Override nc in params if provided
        saqgfl_params['nc'] = nc
        super().__init__(**saqgfl_params)

        if self.debug:
            print(f"\n[SA-QGFL-YOLO] Initialized for {nc} classes (species detection)")
            print(f"  Classes: {[CLASS_NAMES.get(i, f'Class_{i}') for i in range(nc)]}")

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        current_epoch: int = 0,
        total_epochs: int = 200
    ) -> torch.Tensor:
        """
        Compute SA-QGFL loss for YOLO (multi-class)

        YOLO provides soft targets (IoU-quality weighted), not hard 0/1 labels.
        We use per-class alpha/gamma based on species characteristics.

        Args:
            pred_scores: [batch, num_preds, nc] - Predicted logits (NOT probabilities)
            target_scores: [batch, num_preds, nc] - Soft targets ∈ [0, 1] (IoU-weighted)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar SA-QGFL loss value
        """
        if self.debug:
            self.validate_tensors(pred_scores, target_scores, stage="SA-QGFL-YOLO input")

        # Get predicted probabilities from logits
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss
        # For soft targets: pt = pred_prob * target + (1 - pred_prob) * (1 - target)
        pt = pred_prob * target_scores + (1 - pred_prob) * (1 - target_scores)

        # Clamp pt to avoid log(0)
        pt = torch.clamp(pt, min=self.epsilon, max=1.0 - self.epsilon)

        # Get class IDs from targets (multi-class, not binary)
        # For YOLO's soft targets, use argmax to determine dominant class
        class_ids = target_scores.argmax(dim=-1)  # [batch, num_preds]

        # Compute per-class alpha (PWA - Prevalence-Weighted Alpha)
        # Shape: [batch, num_preds] -> expand to [batch, num_preds, nc]
        alpha = self.get_class_alpha_tensor(class_ids)  # [batch, num_preds]
        alpha = alpha.unsqueeze(-1).expand_as(pred_scores)  # [batch, num_preds, nc]

        # Compute effective gamma (MCP + difficulty-aware scaling)
        # Base gamma comes from MCP, then modulated by sample difficulty
        gamma_eff = self._compute_gamma_eff_for_yolo(pt, class_ids)  # [batch, num_preds, nc]

        if self.debug:
            print(f"\n[DEBUG] SA-QGFL-YOLO Forward Pass:")
            print(f"  Batch size: {pred_scores.shape[0]}")
            print(f"  Num predictions: {pred_scores.shape[1]}")
            print(f"  Num classes: {pred_scores.shape[2]}")
            print(f"  Class ID distribution: {self._get_class_distribution(class_ids)}")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff.min():.4f}, {gamma_eff.max():.4f}]")

        # Compute quality weight (using HQS if enabled)
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

        # Apply all SA-QGFL components:
        # SA-QGFL = alpha * (1-pt)^gamma_eff * (1 + quality_weight) * uiou_ratio * BCE
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
            self.check_loss_sanity(loss_total, stage="SA-QGFL-YOLO")

        return loss_total

    def _compute_gamma_eff_for_yolo(
        self,
        pt: torch.Tensor,
        class_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective gamma for YOLO with per-class MCP + difficulty scaling

        This is specialized for YOLO's tensor shapes [batch, num_preds, nc].

        Args:
            pt: Predicted probability for true class [batch, num_preds, nc]
            class_ids: Target class IDs [batch, num_preds]

        Returns:
            gamma_eff: Effective gamma values [batch, num_preds, nc]
        """
        batch_size = pt.shape[0]
        num_preds = pt.shape[1]
        num_classes = pt.shape[2]

        # Get per-class base gamma from MCP
        base_gamma = torch.tensor(2.0, device=pt.device, dtype=pt.dtype)
        max_gamma = self.get_class_gamma_tensor(class_ids)  # [batch, num_preds]
        max_gamma = max_gamma.unsqueeze(-1).expand(batch_size, num_preds, num_classes)

        # Raw difficulty
        raw_difficulty = 1.0 - pt

        # Difficulty thresholding
        difficulty_adjusted = torch.clamp(
            raw_difficulty - self.difficulty_threshold,
            min=0.0
        )

        # Normalize to [0, 1]
        if self.difficulty_threshold < 1.0:
            difficulty = difficulty_adjusted / (1.0 - self.difficulty_threshold + self.epsilon)
        else:
            difficulty = difficulty_adjusted

        # Interpolate between base and max gamma
        gamma_eff = base_gamma + (max_gamma - base_gamma) * difficulty

        return gamma_eff

    def _get_class_distribution(self, class_ids: torch.Tensor) -> Dict[str, int]:
        """Get distribution of classes in current batch (for debugging)"""
        dist = {}
        for c in range(self.nc):
            count = (class_ids == c).sum().item()
            name = CLASS_NAMES.get(c, f"Class_{c}")
            dist[name] = count
        return dist


def create_saqgfl_yolo_loss(
    nc: int = NC,
    use_hqs: bool = True,
    debug: bool = False,
    **kwargs
) -> SAQGFLYOLOLoss:
    """
    Factory function to create SA-QGFL YOLO loss with sensible defaults

    Args:
        nc: Number of classes (default 5 for D3 species)
        use_hqs: Enable Hierarchical Quality Scoring
        debug: Enable debug output
        **kwargs: Additional SA-QGFL parameters

    Returns:
        Configured SAQGFLYOLOLoss instance
    """
    return SAQGFLYOLOLoss(
        nc=nc,
        use_hqs=use_hqs,
        debug=debug,
        **kwargs
    )


# Standalone testing
if __name__ == "__main__":
    print("Testing SA-QGFL YOLO Loss (Multi-Class)...")

    # Initialize for 5-class species detection
    saqgfl_yolo = SAQGFLYOLOLoss(nc=5, debug=True, use_hqs=True)

    # Simulate YOLO outputs
    batch_size = 2
    num_preds = 100  # Simplified (real YOLO has ~8400)
    nc = 5

    print("\n" + "="*60)
    print("Test: SA-QGFL YOLO Forward Pass (5 Classes)")
    print("="*60)

    # Predicted logits (before sigmoid)
    pred_scores = torch.randn(batch_size, num_preds, nc)

    # Soft targets (IoU-quality weighted)
    # Most predictions are uninfected (class 0)
    target_scores = torch.zeros(batch_size, num_preds, nc)
    target_scores[:, :, 0] = torch.rand(batch_size, num_preds) * 0.3 + 0.7  # High uninfected

    # Make some predictions infected (various species)
    # P. falciparum (class 1) - most common parasite
    target_scores[:, :5, 0] = 0.1
    target_scores[:, :5, 1] = 0.9

    # P. ovale (class 2) - rare
    target_scores[:, 5:7, 0] = 0.1
    target_scores[:, 5:7, 2] = 0.9

    # P. malariae (class 3) - rarest
    target_scores[:, 7:8, 0] = 0.1
    target_scores[:, 7:8, 3] = 0.9

    # P. vivax (class 4) - rare
    target_scores[:, 8:10, 0] = 0.1
    target_scores[:, 8:10, 4] = 0.9

    print(f"\nInput shapes:")
    print(f"  pred_scores: {pred_scores.shape}")
    print(f"  target_scores: {target_scores.shape}")

    # Forward pass
    loss = saqgfl_yolo(
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
    loss_grad = saqgfl_yolo(pred_scores_grad, target_scores, current_epoch=50, total_epochs=200)
    loss_grad.backward()

    print(f"Gradients computed: {pred_scores_grad.grad is not None}")
    if pred_scores_grad.grad is not None:
        print(f"Gradient range: [{pred_scores_grad.grad.min():.6f}, {pred_scores_grad.grad.max():.6f}]")
        print(f"Gradient mean: {pred_scores_grad.grad.mean():.6f}")
        print(f"Non-zero gradients: {(pred_scores_grad.grad != 0).sum().item()} / {pred_scores_grad.grad.numel()}")

    # Test without HQS
    print("\n" + "="*60)
    print("Test: Without HQS (Standard Quality)")
    print("="*60)

    saqgfl_yolo_no_hqs = SAQGFLYOLOLoss(nc=5, debug=False, use_hqs=False)
    loss_no_hqs = saqgfl_yolo_no_hqs(pred_scores, target_scores, 50, 200)
    print(f"Loss without HQS: {loss_no_hqs.item():.6f}")

    print("\n" + "="*60)
    print("SA-QGFL YOLO Tests Passed! ✓")
    print("="*60)
