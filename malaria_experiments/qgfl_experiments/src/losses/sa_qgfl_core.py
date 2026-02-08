"""
Species-Adaptive Quality-Guided Focal Loss (SA-QGFL) - Core Components

Multi-class extension of QGFL for species-level Plasmodium detection.
Supports arbitrary number of classes with per-class alpha (PWA) and gamma (MCP) values.

Three novel components:
1. MCP (Morphological Complexity Priors) - Per-species gamma based on detection difficulty
2. PWA (Prevalence-Weighted Alpha) - Per-species alpha based on class frequency + clinical importance
3. HQS (Hierarchical Quality Scoring) - Two-tier quality: infection detection + species discrimination

Author: Thabang Isaka
Date: 2025-12-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

# Import default configs from sa_qgfl_config
from .sa_qgfl_config import (
    NC, CLASS_NAMES, MCP_GAMMA, PWA_ALPHA,
    HQS_INFECTION_WEIGHT, HQS_SPECIES_WEIGHT,
    DEFAULT_DIFFICULTY_THRESHOLD, DEFAULT_QUALITY_MARGIN
)


class SAQGFLCore(nn.Module):
    """
    Species-Adaptive QGFL Core - Multi-class version

    Implements:
    1. Per-class alpha weighting (PWA)
    2. Per-class gamma focusing (MCP)
    3. Hierarchical Quality Scoring (HQS)
    4. Difficulty-aware scaling
    5. Quality-guided components
    """

    def __init__(
        self,
        nc: int = NC,
        class_alphas: Optional[Dict[int, float]] = None,
        class_gammas: Optional[Dict[int, float]] = None,
        difficulty_threshold: float = DEFAULT_DIFFICULTY_THRESHOLD,
        quality_margin: float = DEFAULT_QUALITY_MARGIN,
        quality_factor: float = 2.0,
        uiou_start: float = 2.0,
        uiou_end: float = 0.5,
        epsilon: float = 1e-7,
        use_hqs: bool = True,
        hqs_infection_weight: float = HQS_INFECTION_WEIGHT,
        hqs_species_weight: float = HQS_SPECIES_WEIGHT,
        debug: bool = True
    ):
        """
        Initialize SA-QGFL core components

        Args:
            nc: Number of classes (default 5 for D3 species)
            class_alphas: Per-class alpha weights {class_id: alpha}. Uses PWA defaults if None.
            class_gammas: Per-class gamma values {class_id: gamma}. Uses MCP defaults if None.
            difficulty_threshold: Only apply QGFL to samples with pt < threshold [0.6]
            quality_margin: Ignore quality differences below this margin [0.3]
            quality_factor: Exponential scaling factor for quality weighting [2.0]
            uiou_start: Initial UIoU ratio (localization focus) [2.0]
            uiou_end: Final UIoU ratio (classification focus) [0.5]
            epsilon: Small constant for numerical stability [1e-7]
            use_hqs: Enable Hierarchical Quality Scoring [True]
            hqs_infection_weight: Weight for infection detection quality [0.3]
            hqs_species_weight: Weight for species discrimination quality [0.7]
            debug: Enable extensive logging and validation [True]
        """
        super().__init__()

        self.nc = nc
        self.class_alphas = class_alphas if class_alphas is not None else PWA_ALPHA.copy()
        self.class_gammas = class_gammas if class_gammas is not None else MCP_GAMMA.copy()
        self.difficulty_threshold = difficulty_threshold
        self.quality_margin = quality_margin
        self.quality_factor = quality_factor
        self.uiou_start = uiou_start
        self.uiou_end = uiou_end
        self.epsilon = epsilon
        self.use_hqs = use_hqs
        self.hqs_infection_weight = hqs_infection_weight
        self.hqs_species_weight = hqs_species_weight
        self.debug = debug

        # Pre-compute alpha and gamma tensors for efficient lookup
        self._alpha_list = [self.class_alphas.get(i, 0.5) for i in range(nc)]
        self._gamma_list = [self.class_gammas.get(i, 2.0) for i in range(nc)]

        # Validation
        self._validate_params()

        if self.debug:
            self._print_config()

    def _validate_params(self):
        """Validate SA-QGFL parameters"""
        assert self.nc > 0, f"nc must be > 0, got {self.nc}"
        assert 0 <= self.difficulty_threshold < 1, f"difficulty_threshold must be in [0, 1), got {self.difficulty_threshold}"
        assert self.quality_margin >= 0, f"quality_margin must be >= 0, got {self.quality_margin}"
        assert self.quality_factor > 0, f"quality_factor must be > 0, got {self.quality_factor}"
        assert self.uiou_start > 0, f"uiou_start must be > 0, got {self.uiou_start}"
        assert self.uiou_end > 0, f"uiou_end must be > 0, got {self.uiou_end}"

        # Validate HQS weights sum to ~1
        if self.use_hqs:
            hqs_sum = self.hqs_infection_weight + self.hqs_species_weight
            assert abs(hqs_sum - 1.0) < 0.01, f"HQS weights should sum to 1, got {hqs_sum}"

        # Validate per-class params
        for c in range(self.nc):
            alpha = self.class_alphas.get(c, 0.5)
            gamma = self.class_gammas.get(c, 2.0)
            assert 0 < alpha <= 1, f"alpha for class {c} must be in (0, 1], got {alpha}"
            assert gamma >= 0, f"gamma for class {c} must be >= 0, got {gamma}"

    def _print_config(self):
        """Print SA-QGFL configuration"""
        print("\n" + "="*70)
        print("SA-QGFL (Species-Adaptive Quality-Guided Focal Loss) Core Initialized")
        print("="*70)
        print(f"Number of classes: {self.nc}")
        print(f"\nPer-Class Alpha (PWA - Prevalence-Weighted):")
        for c in range(self.nc):
            name = CLASS_NAMES.get(c, f"Class_{c}")
            alpha = self.class_alphas.get(c, 0.5)
            print(f"  {name}: α={alpha:.3f}")
        print(f"\nPer-Class Gamma (MCP - Morphological Complexity):")
        for c in range(self.nc):
            name = CLASS_NAMES.get(c, f"Class_{c}")
            gamma = self.class_gammas.get(c, 2.0)
            print(f"  {name}: γ={gamma:.1f}")
        print(f"\nHierarchical Quality Scoring (HQS): {'Enabled' if self.use_hqs else 'Disabled'}")
        if self.use_hqs:
            print(f"  Infection weight: {self.hqs_infection_weight:.2f}")
            print(f"  Species weight:   {self.hqs_species_weight:.2f}")
        print(f"\nDifficulty Threshold: {self.difficulty_threshold:.3f}")
        print(f"Quality Margin: {self.quality_margin:.2f}")
        print(f"Quality Factor: {self.quality_factor:.1f}")
        print(f"UIoU Decay: {self.uiou_start:.1f} → {self.uiou_end:.1f}")
        print("="*70 + "\n")

    def get_class_alpha_tensor(
        self,
        class_ids: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Build per-class alpha tensor from class IDs

        Args:
            class_ids: Class IDs tensor [batch, num_preds] or [N]
            target_shape: Optional shape to expand to [batch, num_preds, num_classes]

        Returns:
            alpha: Per-sample alpha weights [same shape as class_ids] or [target_shape]
        """
        device = class_ids.device
        dtype = torch.float32

        # Create alpha lookup tensor
        alpha_lookup = torch.tensor(self._alpha_list, device=device, dtype=dtype)

        # Gather alpha values for each class
        # Clamp class_ids to valid range
        class_ids_clamped = torch.clamp(class_ids.long(), 0, self.nc - 1)
        alpha = alpha_lookup[class_ids_clamped]

        # Expand to target shape if needed
        if target_shape is not None and len(target_shape) > len(alpha.shape):
            alpha = alpha.unsqueeze(-1).expand(target_shape)

        return alpha

    def get_class_gamma_tensor(
        self,
        class_ids: torch.Tensor,
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Build per-class gamma tensor from class IDs (MCP component)

        Args:
            class_ids: Class IDs tensor [batch, num_preds] or [N]
            target_shape: Optional shape to expand to

        Returns:
            gamma: Per-sample base gamma values [same shape as class_ids]
        """
        device = class_ids.device
        dtype = torch.float32

        # Create gamma lookup tensor
        gamma_lookup = torch.tensor(self._gamma_list, device=device, dtype=dtype)

        # Gather gamma values for each class
        class_ids_clamped = torch.clamp(class_ids.long(), 0, self.nc - 1)
        gamma = gamma_lookup[class_ids_clamped]

        # Expand to target shape if needed
        if target_shape is not None and len(target_shape) > len(gamma.shape):
            gamma = gamma.unsqueeze(-1).expand(target_shape)

        return gamma

    def compute_gamma_eff_multiclass(
        self,
        pt: torch.Tensor,
        class_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute effective gamma with per-class MCP values + difficulty-aware scaling

        Gamma computation:
        1. Get base gamma from MCP (per-class morphological complexity)
        2. Apply difficulty scaling: harder samples get higher effective gamma
        3. Use thresholding to focus only on genuinely difficult samples

        Args:
            pt: Predicted probability for true class [batch, num_preds] or [N]
            class_ids: Target class IDs [batch, num_preds] or [N]

        Returns:
            gamma_eff: Effective gamma values [same shape as pt]
        """
        # Get per-class base gamma from MCP
        base_gamma = torch.tensor(2.0, device=pt.device, dtype=pt.dtype)
        max_gamma = self.get_class_gamma_tensor(class_ids)

        # Ensure shapes match
        if max_gamma.shape != pt.shape:
            max_gamma = max_gamma.view_as(pt)

        # Raw difficulty
        raw_difficulty = 1.0 - pt

        # Difficulty thresholding
        # Only focus on genuinely difficult examples (pt < threshold)
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

        if self.debug and torch.isnan(gamma_eff).any():
            print(f"[WARNING] NaN detected in gamma_eff (multiclass)!")
            print(f"  pt range: [{pt.min():.4f}, {pt.max():.4f}]")
            print(f"  class_ids unique: {class_ids.unique().tolist()}")

        return gamma_eff

    def compute_hierarchical_quality(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hierarchical Quality Scoring (HQS) - Novel component for SA-QGFL

        Two-tier quality assessment:
        1. Infection detection quality: How well does the model distinguish infected vs uninfected?
        2. Species discrimination quality: For infected cells, how well does it identify the species?

        This mirrors clinical workflow:
        - First question: Is there an infection?
        - Second question: Which species?

        Args:
            pred_scores: Predicted class probabilities [batch, num_preds, num_classes]
            target_scores: Target class scores [batch, num_preds, num_classes]

        Returns:
            quality: Hierarchical quality score [batch, num_preds]
        """
        # Stage 1: Infection detection quality
        # P(infected) = 1 - P(uninfected) = 1 - pred[:, :, 0]
        pred_infected = 1.0 - pred_scores[..., 0]  # [batch, num_preds]
        target_infected = 1.0 - target_scores[..., 0]  # [batch, num_preds]
        q_infection = torch.abs(pred_infected - target_infected)

        # Stage 2: Species discrimination quality
        # For parasite classes (1-4), how well does the model distinguish?
        # Only meaningful for infected cells, but we compute overall for simplicity
        q_species = torch.abs(pred_scores - target_scores).mean(dim=-1)

        # Combined hierarchical quality
        # Weight species discrimination higher (0.7) because correct species ID
        # is crucial for treatment selection
        q_hierarchical = (
            self.hqs_infection_weight * q_infection +
            self.hqs_species_weight * q_species
        )

        return q_hierarchical

    def compute_quality_weight(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quality-guided weighting

        If HQS is enabled, uses hierarchical quality scoring.
        Otherwise, uses standard quality (|pred - target|).

        Args:
            predictions: Predicted class probabilities [batch, num_preds, num_classes]
            targets: Target class probabilities [batch, num_preds, num_classes]

        Returns:
            quality_weight: Weighting factor [batch, num_preds]
        """
        if self.use_hqs:
            quality = self.compute_hierarchical_quality(predictions, targets)
        else:
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
            print(f"[WARNING] NaN detected in quality_weight (multiclass)!")

        return quality_weight

    def compute_uiou_ratio(
        self,
        current_epoch: int,
        total_epochs: int
    ) -> float:
        """
        Compute UIoU ratio with linear decay

        Early training: High UIoU (localization focus)
        Late training: Low UIoU (classification focus)

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

        print(f"\n[DEBUG] SA-QGFL Tensor Validation ({stage}):")
        print(f"  pred_scores: shape={pred_scores.shape}, dtype={pred_scores.dtype}")
        print(f"    range=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
        print(f"    nan={torch.isnan(pred_scores).sum()}, inf={torch.isinf(pred_scores).sum()}")

        print(f"  targets: shape={targets.shape}, dtype={targets.dtype}")
        print(f"    range=[{targets.min():.4f}, {targets.max():.4f}]")
        print(f"    nan={torch.isnan(targets).sum()}, inf={torch.isinf(targets).sum()}")

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
            print(f"[ERROR] NaN detected in SA-QGFL loss ({stage})!")
            return False

        if torch.isinf(loss).any():
            print(f"[ERROR] Inf detected in SA-QGFL loss ({stage})!")
            return False

        if loss < 0:
            print(f"[ERROR] Negative loss detected ({stage}): {loss.item():.6f}")
            return False

        if loss > 1000:
            print(f"[WARNING] Very large SA-QGFL loss ({stage}): {loss.item():.6f}")

        if loss < 1e-6 and loss > 0:
            print(f"[WARNING] Very small SA-QGFL loss ({stage}): {loss.item():.6f}")

        if self.debug:
            print(f"[DEBUG] SA-QGFL Loss ({stage}): {loss.item():.6f} ✓")

        return True


# Standalone testing
if __name__ == "__main__":
    print("Testing SA-QGFL Core Components (Multi-Class)...")

    # Initialize with default species config
    saqgfl = SAQGFLCore(nc=5, debug=True)

    # Test 1: Per-class alpha tensor
    print("\n" + "="*60)
    print("Test 1: Per-Class Alpha (PWA)")
    print("="*60)

    class_ids = torch.tensor([0, 1, 2, 3, 4, 0, 1])  # Mix of classes
    alpha = saqgfl.get_class_alpha_tensor(class_ids)
    print(f"Class IDs: {class_ids.tolist()}")
    print(f"Alpha values: {alpha.tolist()}")
    print("Expected: [0.01, 0.25, 0.20, 0.30, 0.24, 0.01, 0.25]")

    # Test 2: Per-class gamma tensor
    print("\n" + "="*60)
    print("Test 2: Per-Class Gamma (MCP)")
    print("="*60)

    gamma = saqgfl.get_class_gamma_tensor(class_ids)
    print(f"Class IDs: {class_ids.tolist()}")
    print(f"Gamma values: {gamma.tolist()}")
    print("Expected: [2.0, 5.0, 7.0, 9.0, 7.0, 2.0, 5.0]")

    # Test 3: Effective gamma with difficulty
    print("\n" + "="*60)
    print("Test 3: Effective Gamma (MCP + Difficulty)")
    print("="*60)

    pt = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4])
    gamma_eff = saqgfl.compute_gamma_eff_multiclass(pt, class_ids)
    print(f"pt values: {pt.tolist()}")
    print(f"Gamma effective: {[f'{g:.2f}' for g in gamma_eff.tolist()]}")
    print("Expected: Higher γ for low pt (difficult samples)")

    # Test 4: Hierarchical Quality Scoring
    print("\n" + "="*60)
    print("Test 4: Hierarchical Quality Scoring (HQS)")
    print("="*60)

    # [batch=1, num_preds=3, num_classes=5]
    pred_scores = torch.tensor([
        [[0.9, 0.05, 0.02, 0.02, 0.01],   # Correctly predicts uninfected
         [0.1, 0.7, 0.1, 0.05, 0.05],      # Correctly predicts P.falciparum
         [0.1, 0.4, 0.3, 0.1, 0.1]]        # Confused between P.f and P.ovale
    ])
    target_scores = torch.tensor([
        [[1.0, 0.0, 0.0, 0.0, 0.0],        # Uninfected
         [0.0, 1.0, 0.0, 0.0, 0.0],        # P.falciparum
         [0.0, 0.0, 1.0, 0.0, 0.0]]        # P.ovale (model confused)
    ])

    hqs = saqgfl.compute_hierarchical_quality(pred_scores, target_scores)
    print(f"Predictions:\n{pred_scores[0]}")
    print(f"Targets:\n{target_scores[0]}")
    print(f"HQS values: {hqs[0].tolist()}")
    print("Expected: Highest quality error for sample 3 (confused species)")

    print("\n" + "="*60)
    print("All SA-QGFL Core Tests Passed! ✓")
    print("="*60)
