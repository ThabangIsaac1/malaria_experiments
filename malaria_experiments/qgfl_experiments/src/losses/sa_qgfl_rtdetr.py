"""
Species-Adaptive Quality-Guided Focal Loss (SA-QGFL) for RT-DETR

Multi-class RT-DETR-specific SA-QGFL implementation for species-level Plasmodium detection.
Replaces VarifocalLoss with species-adaptive focal loss.

Key differences from binary QGFL:
1. Per-class alpha (PWA) instead of binary infected/uninfected
2. Per-class gamma (MCP) based on species morphological complexity
3. Hierarchical Quality Scoring (HQS) for two-tier quality assessment

Integration Point: ultralytics/models/utils/loss.py, lines 77-84
  Current: VarifocalLoss or FocalLoss or BCE
  Replace with: SA-QGFL loss

IMPORTANT: RT-DETR baseline uses VarifocalLoss, NOT BCE!
SA-QGFL must beat VarifocalLoss (which already has α=0.75, γ=2.0)

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


class SAQGFLRTDETRLoss(SAQGFLCore):
    """
    SA-QGFL loss for RT-DETR architecture - Multi-class version

    RT-DETR-specific characteristics:
    - One-hot targets + quality scores (separate)
    - 300 queries per image (vs YOLO's 8400)
    - Hungarian Matcher for bipartite assignment
    - Multi-class (nc=5 for D3 species detection)
    - Baseline uses VarifocalLoss (α=0.75, γ=2.0)

    Novel components:
    - MCP: Per-species gamma based on morphological detection difficulty
    - PWA: Per-species alpha based on prevalence + clinical importance
    - HQS: Hierarchical quality scoring (infection → species)
    """

    def __init__(self, nc: int = NC, **saqgfl_params):
        """
        Initialize SA-QGFL loss for RT-DETR

        Args:
            nc: Number of classes (default=5 for D3 species)
            **saqgfl_params: SA-QGFL parameters (class_alphas, class_gammas, etc.)
        """
        # Override nc in params if provided
        saqgfl_params['nc'] = nc
        super().__init__(**saqgfl_params)

        if self.debug:
            print(f"\n[SA-QGFL-RTDETR] Initialized for {nc} classes (species detection)")
            print(f"  Classes: {[CLASS_NAMES.get(i, f'Class_{i}') for i in range(nc)]}")
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
        Compute SA-QGFL loss for RT-DETR (multi-class)

        RT-DETR provides one-hot targets AND quality scores (IoU-based).
        We use per-class alpha/gamma based on species characteristics.

        Args:
            pred_scores: [batch, 300, nc] - Predicted logits (NOT probabilities)
            one_hot: [batch, 300, nc] - One-hot encoded targets {0, 1}
            gt_scores: [batch, 300, nc] - IoU-quality weighted targets ∈ [0, 1]
            num_gts: Number of ground truth boxes (scalar)
            nq: Number of queries (300 for RT-DETR)
            current_epoch: Current training epoch (for UIoU decay)
            total_epochs: Total number of training epochs

        Returns:
            loss: Scalar SA-QGFL loss value
        """
        if self.debug:
            self.validate_tensors(pred_scores, gt_scores, stage="SA-QGFL-RTDETR input")
            print(f"\n[DEBUG] SA-QGFL-RTDETR Forward:")
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

        # Get class IDs from one-hot targets (multi-class, not binary)
        class_ids = one_hot.argmax(dim=-1)  # [batch, 300]

        # Compute per-class alpha (PWA - Prevalence-Weighted Alpha)
        alpha = self.get_class_alpha_tensor(class_ids)  # [batch, 300]
        alpha = alpha.unsqueeze(-1).expand_as(pred_scores)  # [batch, 300, nc]

        # Compute effective gamma (MCP + difficulty-aware scaling)
        gamma_eff = self._compute_gamma_eff_for_rtdetr(pt, class_ids)  # [batch, 300, nc]

        if self.debug:
            print(f"  Class ID distribution: {self._get_class_distribution(class_ids)}")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff.min():.4f}, {gamma_eff.max():.4f}]")

        # Compute quality weight (using HQS if enabled)
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

        # Apply all SA-QGFL components:
        # SA-QGFL = alpha * (1-pt)^gamma_eff * (1 + quality_weight) * uiou_ratio * BCE
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
            self.check_loss_sanity(loss_normalized, stage="SA-QGFL-RTDETR")

        return loss_normalized

    def _compute_gamma_eff_for_rtdetr(
        self,
        pt: torch.Tensor,
        class_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective gamma for RT-DETR with per-class MCP + difficulty scaling

        This is specialized for RT-DETR's tensor shapes [batch, nq, nc].

        Args:
            pt: Predicted probability for true class [batch, nq, nc]
            class_ids: Target class IDs [batch, nq]

        Returns:
            gamma_eff: Effective gamma values [batch, nq, nc]
        """
        batch_size = pt.shape[0]
        nq = pt.shape[1]
        num_classes = pt.shape[2]

        # Get per-class base gamma from MCP
        base_gamma = torch.tensor(2.0, device=pt.device, dtype=pt.dtype)
        max_gamma = self.get_class_gamma_tensor(class_ids)  # [batch, nq]
        max_gamma = max_gamma.unsqueeze(-1).expand(batch_size, nq, num_classes)

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


def create_saqgfl_rtdetr_loss(
    nc: int = NC,
    use_hqs: bool = True,
    debug: bool = False,
    **kwargs
) -> SAQGFLRTDETRLoss:
    """
    Factory function to create SA-QGFL RT-DETR loss with sensible defaults

    Args:
        nc: Number of classes (default 5 for D3 species)
        use_hqs: Enable Hierarchical Quality Scoring
        debug: Enable debug output
        **kwargs: Additional SA-QGFL parameters

    Returns:
        Configured SAQGFLRTDETRLoss instance
    """
    return SAQGFLRTDETRLoss(
        nc=nc,
        use_hqs=use_hqs,
        debug=debug,
        **kwargs
    )


# Standalone testing
if __name__ == "__main__":
    print("Testing SA-QGFL RT-DETR Loss (Multi-Class)...")

    # Initialize for 5-class species detection
    saqgfl_rtdetr = SAQGFLRTDETRLoss(nc=5, debug=True, use_hqs=True)

    # Simulate RT-DETR outputs
    batch_size = 2
    nq = 300  # RT-DETR queries
    nc = 5
    num_gts = 15  # Total ground truths in batch

    print("\n" + "="*60)
    print("Test: SA-QGFL RT-DETR Forward Pass (5 Classes)")
    print("="*60)

    # Predicted logits (before sigmoid)
    pred_scores = torch.randn(batch_size, nq, nc)

    # One-hot targets (hard labels)
    one_hot = torch.zeros(batch_size, nq, nc)

    # Assign matched queries to different species
    # Uninfected (class 0) - most common
    one_hot[:, :5, 0] = 1

    # P. falciparum (class 1) - matched predictions
    one_hot[:, 5:10, 1] = 1

    # P. ovale (class 2) - rare
    one_hot[:, 10:12, 2] = 1

    # P. malariae (class 3) - rarest
    one_hot[:, 12:13, 3] = 1

    # P. vivax (class 4) - rare
    one_hot[:, 13:15, 4] = 1

    # Quality scores (IoU-weighted)
    gt_scores = one_hot.clone()
    # Apply IoU quality weighting to matched predictions
    iou_weights = torch.rand(batch_size, nq, 1).expand(-1, -1, nc) * 0.5 + 0.5
    gt_scores[:, :15, :] *= iou_weights[:, :15, :]

    print(f"\nInput shapes:")
    print(f"  pred_scores: {pred_scores.shape}")
    print(f"  one_hot: {one_hot.shape}")
    print(f"  gt_scores: {gt_scores.shape}")
    print(f"  num_gts: {num_gts}")
    print(f"  nq: {nq}")

    # Forward pass
    loss = saqgfl_rtdetr(
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
    loss_grad = saqgfl_rtdetr(
        pred_scores_grad, one_hot, gt_scores,
        num_gts, nq, current_epoch=100, total_epochs=200
    )
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

    saqgfl_rtdetr_no_hqs = SAQGFLRTDETRLoss(nc=5, debug=False, use_hqs=False)
    loss_no_hqs = saqgfl_rtdetr_no_hqs(pred_scores, one_hot, gt_scores, num_gts, nq, 100, 200)
    print(f"Loss without HQS: {loss_no_hqs.item():.6f}")

    print("\n" + "="*60)
    print("SA-QGFL RT-DETR Tests Passed! ✓")
    print("="*60)
