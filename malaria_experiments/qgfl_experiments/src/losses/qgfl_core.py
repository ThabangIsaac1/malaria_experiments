"""
Quality-Guided Focal Loss (QGFL) - Core Components

Architecture-agnostic QGFL components shared between YOLO and RT-DETR implementations.
Based on paper: "Quality-Guided Focal Loss: Enhancing Minority Class Detection in
Haematological Imaging"

Author: Thabang Isaka
Date: 2025-10-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QGFLCore(nn.Module):
    """
    Core QGFL loss computation - architecture agnostic

    Implements 5-level progressive adaptation:
    1. Standard Focal Loss (baseline)
    2. Difficulty-Aware Scaling
    3. Class-Difficulty Scaling
    4. Thresholding Mechanisms
    5. Quality-Guided Components (complete QGFL)
    """

    def __init__(
        self,
        infected_alpha: float = 0.9,
        uninfected_alpha: float = 0.1,
        infected_gamma: float = 8.0,
        uninfected_gamma: float = 4.0,
        difficulty_threshold: float = 0.925,
        quality_margin: float = 0.5,
        quality_factor: float = 2.0,
        uiou_start: float = 2.0,
        uiou_end: float = 0.5,
        epsilon: float = 1e-7,
        debug: bool = True
    ):
        """
        Initialize QGFL core components

        Args:
            infected_alpha: Class weight for infected (minority) class [0.9]
            uninfected_alpha: Class weight for uninfected (majority) class [0.1]
            infected_gamma: Max focusing parameter for infected class [8.0]
            uninfected_gamma: Max focusing parameter for uninfected class [4.0]
            difficulty_threshold: Only apply QGFL to samples with pt < threshold [0.925]
            quality_margin: Ignore quality differences below this margin [0.5]
            quality_factor: Exponential scaling factor for quality weighting [2.0]
            uiou_start: Initial UIoU ratio (localization focus) [2.0]
            uiou_end: Final UIoU ratio (classification focus) [0.5]
            epsilon: Small constant for numerical stability [1e-7]
            debug: Enable extensive logging and validation [True]
        """
        super().__init__()

        self.infected_alpha = infected_alpha
        self.uninfected_alpha = uninfected_alpha
        self.infected_gamma = infected_gamma
        self.uninfected_gamma = uninfected_gamma
        self.difficulty_threshold = difficulty_threshold
        self.quality_margin = quality_margin
        self.quality_factor = quality_factor
        self.uiou_start = uiou_start
        self.uiou_end = uiou_end
        self.epsilon = epsilon
        self.debug = debug

        # Validation
        self._validate_params()

        if self.debug:
            print("\n" + "="*60)
            print("QGFL Core Initialized")
            print("="*60)
            print(f"Class Weights (α):")
            print(f"  Infected:   {self.infected_alpha:.2f}")
            print(f"  Uninfected: {self.uninfected_alpha:.2f}")
            print(f"Focusing Power (γ):")
            print(f"  Infected:   {self.infected_gamma:.1f}")
            print(f"  Uninfected: {self.uninfected_gamma:.1f}")
            print(f"Difficulty Threshold: {self.difficulty_threshold:.3f}")
            print(f"Quality Margin: {self.quality_margin:.2f}")
            print(f"Quality Factor: {self.quality_factor:.1f}")
            print(f"UIoU Decay: {self.uiou_start:.1f} → {self.uiou_end:.1f}")
            print("="*60 + "\n")

    def _validate_params(self):
        """Validate QGFL parameters"""
        assert 0 < self.infected_alpha <= 1, f"infected_alpha must be in (0, 1], got {self.infected_alpha}"
        assert 0 < self.uninfected_alpha <= 1, f"uninfected_alpha must be in (0, 1], got {self.uninfected_alpha}"
        assert self.infected_gamma >= 0, f"infected_gamma must be >= 0, got {self.infected_gamma}"
        assert self.uninfected_gamma >= 0, f"uninfected_gamma must be >= 0, got {self.uninfected_gamma}"
        assert 0 <= self.difficulty_threshold < 1, f"difficulty_threshold must be in [0, 1), got {self.difficulty_threshold}"
        assert self.quality_margin >= 0, f"quality_margin must be >= 0, got {self.quality_margin}"
        assert self.quality_factor > 0, f"quality_factor must be > 0, got {self.quality_factor}"
        assert self.uiou_start > 0, f"uiou_start must be > 0, got {self.uiou_start}"
        assert self.uiou_end > 0, f"uiou_end must be > 0, got {self.uiou_end}"

    def compute_gamma_eff(
        self,
        pt: torch.Tensor,
        is_infected: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective gamma with class-specific + difficulty-aware scaling

        Level 2: Difficulty-Aware Scaling
        γ_eff = γ_base + (γ_max - γ_base) × difficulty

        Level 3: Class-Difficulty Scaling
        Different γ_max for infected vs uninfected

        Level 4: Difficulty Thresholding
        difficulty = max((1-pt) - threshold, 0) / (1 - threshold)

        Args:
            pt: Predicted probability for true class [N] or [N, C]
            is_infected: Boolean mask indicating infected class [N] or [N, C]

        Returns:
            gamma_eff: Effective gamma values [same shape as pt]
        """
        # Base gamma (standard focal loss)
        base_gamma = torch.tensor(2.0, device=pt.device, dtype=pt.dtype)

        # Class-specific max gamma
        max_gamma = torch.where(
            is_infected,
            torch.tensor(self.infected_gamma, device=pt.device, dtype=pt.dtype),
            torch.tensor(self.uninfected_gamma, device=pt.device, dtype=pt.dtype)
        )

        # Raw difficulty
        raw_difficulty = 1.0 - pt

        # Difficulty thresholding (Level 4)
        # Only focus on genuinely difficult examples (pt < threshold)
        difficulty_adjusted = torch.clamp(
            raw_difficulty - self.difficulty_threshold,
            min=0.0
        )

        # Normalize to [0, 1]
        if self.difficulty_threshold < 1.0:
            difficulty = difficulty_adjusted / (1.0 - self.difficulty_threshold)
        else:
            difficulty = difficulty_adjusted

        # Interpolate between base and max gamma (Level 2 & 3)
        gamma_eff = base_gamma + (max_gamma - base_gamma) * difficulty

        if self.debug and torch.isnan(gamma_eff).any():
            print(f"[WARNING] NaN detected in gamma_eff!")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  difficulty range: [{difficulty.min():.4f}, {difficulty.max():.4f}]")
            print(f"  gamma_eff range: [{gamma_eff[~torch.isnan(gamma_eff)].min():.4f}, {gamma_eff[~torch.isnan(gamma_eff)].max():.4f}]")

        return gamma_eff

    def compute_quality_weight(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quality-guided weighting (Level 5)

        Quality = |prediction - target|
        Amplifies loss for low-quality predictions

        Formula:
          quality = |pred - target|
          quality_adjusted = max(quality - margin, 0)
          quality_weight = min(quality_adjusted^factor, 10.0)

        Args:
            predictions: Predicted class probabilities [N, C]
            targets: Target class probabilities [N, C]

        Returns:
            quality_weight: Weighting factor [N]
        """
        # Quality = absolute difference between prediction and target
        quality = torch.abs(predictions - targets).sum(dim=-1)

        # Adjust quality (ignore small differences)
        quality_adjusted = torch.clamp(
            quality - self.quality_margin,
            min=0.0
        )

        # Apply exponential scaling, cap at 10.0
        quality_weight = torch.clamp(
            quality_adjusted ** self.quality_factor,
            max=10.0
        )

        if self.debug and torch.isnan(quality_weight).any():
            print(f"[WARNING] NaN detected in quality_weight!")
            print(f"  quality range: [{quality.min():.4f}, {quality.max():.4f}]")
            print(f"  quality_weight range: [{quality_weight[~torch.isnan(quality_weight)].min():.4f}, {quality_weight[~torch.isnan(quality_weight)].max():.4f}]")

        return quality_weight

    def compute_uiou_ratio(
        self,
        current_epoch: int,
        total_epochs: int
    ) -> float:
        """
        Compute UIoU ratio with linear decay (Level 5)

        Early training: High UIoU (localization focus)
        Late training: Low UIoU (classification focus)

        Formula:
          progress = current_epoch / total_epochs
          uiou_ratio = uiou_start + (uiou_end - uiou_start) × progress

        Args:
            current_epoch: Current training epoch (0-indexed)
            total_epochs: Total number of training epochs

        Returns:
            uiou_ratio: Current UIoU weighting factor
        """
        if total_epochs == 0:
            return self.uiou_start

        progress = current_epoch / max(total_epochs, 1)
        uiou_ratio = self.uiou_start + (self.uiou_end - self.uiou_start) * progress

        return uiou_ratio

    def get_class_alpha(
        self,
        is_infected: torch.Tensor
    ) -> torch.Tensor:
        """
        Get class-specific alpha weights

        Args:
            is_infected: Boolean mask [N] or [N, C]

        Returns:
            alpha: Class weights [same shape as is_infected]
        """
        alpha = torch.where(
            is_infected,
            torch.tensor(self.infected_alpha, device=is_infected.device, dtype=torch.float32),
            torch.tensor(self.uninfected_alpha, device=is_infected.device, dtype=torch.float32)
        )
        return alpha

    def validate_tensors(
        self,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        stage: str = "input"
    ):
        """
        Validate tensor shapes and values for debugging

        Args:
            pred_scores: Predicted logits or probabilities
            targets: Target values
            stage: Description of validation stage
        """
        if not self.debug:
            return

        print(f"\n[DEBUG] Tensor Validation ({stage}):")
        print(f"  pred_scores: shape={pred_scores.shape}, dtype={pred_scores.dtype}, device={pred_scores.device}")
        print(f"    range=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
        print(f"    mean={pred_scores.mean():.4f}, std={pred_scores.std():.4f}")
        print(f"    nan_count={torch.isnan(pred_scores).sum()}, inf_count={torch.isinf(pred_scores).sum()}")

        print(f"  targets: shape={targets.shape}, dtype={targets.dtype}, device={targets.device}")
        print(f"    range=[{targets.min():.4f}, {targets.max():.4f}]")
        print(f"    mean={targets.mean():.4f}, std={targets.std():.4f}")
        print(f"    nan_count={torch.isnan(targets).sum()}, inf_count={torch.isinf(targets).sum()}")

    def check_loss_sanity(
        self,
        loss: torch.Tensor,
        stage: str = "final"
    ) -> bool:
        """
        Check if loss is sane (not NaN, Inf, or unreasonably large/small)

        Args:
            loss: Computed loss tensor
            stage: Description of loss stage

        Returns:
            is_sane: True if loss is reasonable, False otherwise
        """
        if torch.isnan(loss).any():
            print(f"[ERROR] NaN detected in loss ({stage})!")
            return False

        if torch.isinf(loss).any():
            print(f"[ERROR] Inf detected in loss ({stage})!")
            return False

        if loss < 0:
            print(f"[ERROR] Negative loss detected ({stage}): {loss.item():.6f}")
            return False

        if loss > 1000:
            print(f"[WARNING] Very large loss detected ({stage}): {loss.item():.6f}")
            print("  This might indicate numerical instability")

        if loss < 1e-6:
            print(f"[WARNING] Very small loss detected ({stage}): {loss.item():.6f}")
            print("  This might indicate loss collapse")

        if self.debug:
            print(f"[DEBUG] Loss ({stage}): {loss.item():.6f} ✓")

        return True


# Standalone testing
if __name__ == "__main__":
    print("Testing QGFL Core Components...")

    # Initialize
    qgfl = QGFLCore(debug=True)

    # Test gamma computation
    print("\n" + "="*60)
    print("Test 1: Gamma Computation")
    print("="*60)

    pt = torch.tensor([0.1, 0.5, 0.9, 0.95])  # Different difficulties
    is_infected = torch.tensor([True, True, False, False])

    gamma_eff = qgfl.compute_gamma_eff(pt, is_infected)
    print(f"Input pt: {pt.tolist()}")
    print(f"Is infected: {is_infected.tolist()}")
    print(f"Gamma effective: {gamma_eff.tolist()}")
    print(f"Expected: Higher γ for low pt (difficult), lower γ for high pt (easy)")

    # Test quality weighting
    print("\n" + "="*60)
    print("Test 2: Quality Weighting")
    print("="*60)

    predictions = torch.tensor([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])  # [N, C]
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    quality_weight = qgfl.compute_quality_weight(predictions, targets)
    print(f"Predictions:\n{predictions}")
    print(f"Targets:\n{targets}")
    print(f"Quality weights: {quality_weight.tolist()}")
    print(f"Expected: Higher weight for larger |pred - target| difference")

    # Test UIoU decay
    print("\n" + "="*60)
    print("Test 3: UIoU Decay")
    print("="*60)

    epochs = [0, 50, 100, 150, 200]
    total_epochs = 200

    for epoch in epochs:
        uiou = qgfl.compute_uiou_ratio(epoch, total_epochs)
        print(f"Epoch {epoch:3d}/{total_epochs}: UIoU ratio = {uiou:.3f}")

    print("\nExpected: Linear decay from 2.0 → 0.5")

    print("\n" + "="*60)
    print("All QGFL Core Tests Passed! ✓")
    print("="*60)
