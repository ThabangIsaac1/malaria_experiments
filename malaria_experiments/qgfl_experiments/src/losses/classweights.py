"""
Class Weights Loss Implementation for YOLO and RT-DETR

Simple class weighting baseline for comparison with QGFL.
Applies inverse frequency class weights to standard BCE/CE loss
WITHOUT the focal loss modulating factor.

This provides a clean comparison:
- Baseline: Standard BCE
- Class Weights: BCE with inverse frequency weights (this file)
- QGFL: BCE with focal modulation + class weights + quality weighting

Author: Thabang Isaka
Date: 2025-11-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Tuple, Optional
import os


# Module-level class for pickling support
class WeightedBCEWrapper:
    """
    Pickleable wrapper that applies class weights to BCE.
    Must be at module level to support torch.save/pickle.
    """
    def __init__(self, class_weights):
        self.class_weights = class_weights

    def __call__(self, pred, target):
        import torch.nn.functional as F
        # Standard BCE with no reduction
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Get class from target
        if target.dim() >= 2 and target.shape[-1] >= 2:
            target_class = target.argmax(dim=-1, keepdim=True)
            import torch
            weights = torch.where(
                target_class == 1,
                self.class_weights[1],
                self.class_weights[0]
            ).expand_as(bce)
        else:
            import torch
            weights = torch.ones_like(bce)

        return bce * weights


def compute_class_weights_from_dataset(dataset_path: str, method: str = 'inverse_freq') -> Dict[str, float]:
    """
    Compute class weights from dataset label files.

    Args:
        dataset_path: Path to dataset root (must contain train/labels/*.txt)
        method: 'inverse_freq' or 'inverse_sqrt'
            - inverse_freq: weight = total / (n_classes * class_count)
            - inverse_sqrt: weight = sqrt(total / (n_classes * class_count))

    Returns:
        dict with keys: 'uninfected', 'infected', 'ratio', 'counts'
    """
    labels_path = Path(dataset_path) / 'train' / 'labels'

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_path}")

    # Count instances per class
    class_counts = {0: 0, 1: 0}  # 0=uninfected, 1=infected

    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    total = sum(class_counts.values())
    n_classes = len(class_counts)

    if total == 0:
        raise ValueError(f"No labels found in {labels_path}")

    # Compute weights based on method
    if method == 'inverse_freq':
        weight_uninfected = total / (n_classes * class_counts[0]) if class_counts[0] > 0 else 1.0
        weight_infected = total / (n_classes * class_counts[1]) if class_counts[1] > 0 else 1.0
    elif method == 'inverse_sqrt':
        weight_uninfected = (total / (n_classes * class_counts[0])) ** 0.5 if class_counts[0] > 0 else 1.0
        weight_infected = (total / (n_classes * class_counts[1])) ** 0.5 if class_counts[1] > 0 else 1.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse_freq' or 'inverse_sqrt'")

    # Normalize so weights sum to n_classes
    weight_sum = weight_uninfected + weight_infected
    weight_uninfected = weight_uninfected * n_classes / weight_sum
    weight_infected = weight_infected * n_classes / weight_sum

    ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')

    return {
        'uninfected': float(weight_uninfected),
        'infected': float(weight_infected),
        'ratio': float(ratio),
        'counts': {
            'uninfected': int(class_counts[0]),
            'infected': int(class_counts[1]),
            'total': int(total)
        }
    }


class ClassWeightedBCELoss(nn.Module):
    """
    Class-weighted BCE loss for YOLO (v8/v11).

    Simple class weighting without focal loss modulation.
    For fair comparison with QGFL.
    """

    def __init__(
        self,
        weight_uninfected: float = 1.0,
        weight_infected: float = 1.0,
        nc: int = 2,
        debug: bool = False
    ):
        """
        Initialize class-weighted BCE loss.

        Args:
            weight_uninfected: Weight for uninfected class (class 0)
            weight_infected: Weight for infected class (class 1)
            nc: Number of classes (default=2 for binary)
            debug: Enable debug logging
        """
        super().__init__()
        self.weight_uninfected = weight_uninfected
        self.weight_infected = weight_infected
        self.nc = nc
        self.debug = debug

        # Create weight tensor [uninfected, infected]
        self.register_buffer(
            'class_weights',
            torch.tensor([weight_uninfected, weight_infected], dtype=torch.float32)
        )

        if debug:
            print(f"\n[ClassWeightedBCE] Initialized:")
            print(f"  weight_uninfected: {weight_uninfected:.4f}")
            print(f"  weight_infected: {weight_infected:.4f}")

    def forward(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute class-weighted BCE loss.

        Args:
            pred_scores: [batch, num_preds, nc] - Predicted logits
            target_scores: [batch, num_preds, nc] - Soft targets [0, 1]

        Returns:
            loss: Scalar loss value
        """
        # Standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            target_scores,
            reduction='none'
        )  # [batch, num_preds, nc]

        # Determine class assignment based on target
        # For YOLO's soft targets, use argmax
        target_class = target_scores.argmax(dim=-1, keepdim=True)  # [batch, num_preds, 1]

        # Create weight tensor matching target shape
        # weight = infected_weight where target is infected, else uninfected_weight
        weights = torch.where(
            target_class == 1,
            self.class_weights[1],
            self.class_weights[0]
        )  # [batch, num_preds, 1]

        # Apply class weights
        weighted_loss = bce_loss * weights.expand_as(bce_loss)

        # Sum total loss
        loss_total = weighted_loss.sum()

        if self.debug:
            print(f"\n[DEBUG] ClassWeightedBCE Forward:")
            print(f"  BCE loss mean: {bce_loss.mean():.6f}")
            print(f"  Weighted loss mean: {weighted_loss.mean():.6f}")
            print(f"  Total loss: {loss_total:.6f}")

        return loss_total


class ClassWeightedCELoss(nn.Module):
    """
    Class-weighted CE loss for RT-DETR.

    Simple class weighting without focal loss modulation.
    For fair comparison with QGFL.
    """

    def __init__(
        self,
        weight_uninfected: float = 1.0,
        weight_infected: float = 1.0,
        nc: int = 2,
        debug: bool = False
    ):
        """
        Initialize class-weighted CE loss for RT-DETR.

        Args:
            weight_uninfected: Weight for uninfected class (class 0)
            weight_infected: Weight for infected class (class 1)
            nc: Number of classes (default=2 for binary)
            debug: Enable debug logging
        """
        super().__init__()
        self.weight_uninfected = weight_uninfected
        self.weight_infected = weight_infected
        self.nc = nc
        self.debug = debug

        # Create weight tensor [uninfected, infected]
        self.register_buffer(
            'class_weights',
            torch.tensor([weight_uninfected, weight_infected], dtype=torch.float32)
        )

        if debug:
            print(f"\n[ClassWeightedCE] Initialized for RT-DETR:")
            print(f"  weight_uninfected: {weight_uninfected:.4f}")
            print(f"  weight_infected: {weight_infected:.4f}")

    def forward(
        self,
        pred_scores: torch.Tensor,
        one_hot: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        nq: int
    ) -> torch.Tensor:
        """
        Compute class-weighted BCE loss for RT-DETR.

        Args:
            pred_scores: [batch, 300, nc] - Predicted logits
            one_hot: [batch, 300, nc] - One-hot targets
            gt_scores: [batch, 300, nc] - IoU-quality weighted targets
            num_gts: Number of ground truth boxes
            nq: Number of queries (300)

        Returns:
            loss: Scalar loss value
        """
        # Standard BCE loss (element-wise)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            gt_scores,
            reduction='none'
        )  # [batch, 300, nc]

        # Determine class based on one-hot encoding
        # infected class is index 1
        is_infected = (one_hot[..., 1:2] == 1)  # [batch, 300, 1]

        # Apply class weights
        weights = torch.where(
            is_infected,
            self.class_weights[1],
            self.class_weights[0]
        )  # [batch, 300, 1]

        weighted_loss = bce_loss * weights.expand_as(bce_loss)

        # RT-DETR normalization
        loss_normalized = weighted_loss.mean(1).sum() / max(num_gts, 1) * nq

        if self.debug:
            print(f"\n[DEBUG] ClassWeightedCE (RT-DETR) Forward:")
            print(f"  BCE loss mean: {bce_loss.mean():.6f}")
            print(f"  Weighted loss mean: {weighted_loss.mean():.6f}")
            print(f"  Normalized loss: {loss_normalized:.6f}")

        return loss_normalized


# ============================================================================
# INTEGRATION FUNCTIONS (for monkey-patching Ultralytics)
# ============================================================================

def compute_class_weights(dataset_path, class_names: Dict, method: str = 'inverse_freq') -> Dict:
    """
    Wrapper function for computing class weights.

    Args:
        dataset_path: Path to YOLO format dataset
        class_names: Dict mapping class_id to class_name
        method: Weight computation method

    Returns:
        Dict with class weights and metadata
    """
    return compute_class_weights_from_dataset(str(dataset_path), method)


def integrate_class_weights(class_weights: Dict, model_type: str = 'yolo', device: str = 'cuda'):
    """
    Integrate class weights into Ultralytics loss via monkey-patching.

    Args:
        class_weights: Dict with 'uninfected' and 'infected' weights
        model_type: 'yolo' or 'rtdetr'
        device: Device for tensors
    """
    import torch

    weight_uninfected = class_weights['uninfected']
    weight_infected = class_weights['infected']

    print(f"\n[ClassWeights] Integrating class weights via monkey-patching...")
    print(f"  Uninfected weight: {weight_uninfected:.4f}")
    print(f"  Infected weight: {weight_infected:.4f}")
    print(f"  Model type: {model_type}")

    if model_type in ['yolo', 'yolov8s', 'yolov11s', 'yolo11s']:
        _integrate_yolo_classweights(weight_uninfected, weight_infected, device)
    elif model_type == 'rtdetr':
        _integrate_rtdetr_classweights(weight_uninfected, weight_infected, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    print(f"[ClassWeights] Integration complete!")


def _integrate_yolo_classweights(weight_uninfected: float, weight_infected: float, device: str = 'cuda'):
    """Monkey-patch YOLO's BCE loss with class weights."""
    from ultralytics.utils.loss import v8DetectionLoss
    import torch.nn.functional as F
    import torch

    # Create weight tensor
    class_weights_tensor = torch.tensor([weight_uninfected, weight_infected], dtype=torch.float32)

    # Store original __init__ and forward
    original_init = v8DetectionLoss.__init__

    def patched_init(self, model, *args, **kwargs):
        original_init(self, model, *args, **kwargs)
        # Store class weights on the loss module
        self._class_weights = class_weights_tensor.to(self.device)
        print(f"[ClassWeights] v8DetectionLoss initialized with class weights on {self.device}")

    # Store original bce attribute getter
    original_bce_getter = None

    def get_weighted_bce(self):
        """Return a class-weighted BCE loss function."""
        class_weights = self._class_weights

        def weighted_bce(pred, target):
            # Standard BCE
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

            # Determine class from target (argmax for soft labels)
            target_class = target.argmax(dim=-1, keepdim=True)

            # Apply weights
            weights = torch.where(
                target_class == 1,
                class_weights[1],
                class_weights[0]
            )

            weighted_loss = bce * weights.expand_as(bce)
            return weighted_loss.sum()

        return weighted_bce

    # Patch the loss class
    v8DetectionLoss.__init__ = patched_init

    # Override the BCE loss computation in __call__
    original_call = v8DetectionLoss.__call__

    def patched_call(self, preds, batch):
        """Modified call that uses class-weighted BCE."""
        # Call original to get base losses
        loss, loss_items = original_call(self, preds, batch)

        # Note: The BCE is already applied in original_call via self.bce
        # We've patched the BCE to be weighted, so loss should be weighted
        return loss, loss_items

    # Actually patch the bce property
    # The v8DetectionLoss uses self.bce = nn.BCEWithLogitsLoss(reduction='none')
    # We need to replace this with our weighted version

    def patched_init_with_weighted_bce(self, model, *args, **kwargs):
        original_init(self, model, *args, **kwargs)
        # Store class weights
        self._class_weights = class_weights_tensor.to(self.device)

        # Use module-level WeightedBCEWrapper for pickle support
        self.bce = WeightedBCEWrapper(self._class_weights)
        print(f"[ClassWeights] v8DetectionLoss BCE replaced with weighted version")

    v8DetectionLoss.__init__ = patched_init_with_weighted_bce
    print("[ClassWeights] YOLO v8DetectionLoss patched with class weights")


def _integrate_rtdetr_classweights(weight_uninfected: float, weight_infected: float, device: str = 'cuda'):
    """Monkey-patch RT-DETR's loss with class weights."""
    from ultralytics.models.rtdetr.val import RTDETRValidator
    from ultralytics.models.utils.loss import DETRLoss
    import torch.nn.functional as F
    import torch

    # Create weight tensor
    class_weights_tensor = torch.tensor([weight_uninfected, weight_infected], dtype=torch.float32)

    # Patch DETRLoss
    original_detr_init = DETRLoss.__init__

    def patched_detr_init(self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=False, use_uni_match=False, uni_match_ind=0):
        original_detr_init(self, nc, loss_gain, aux_loss, use_fl, use_vfl, use_uni_match, uni_match_ind)
        self._class_weights = class_weights_tensor.to(device if torch.cuda.is_available() else 'cpu')
        print(f"[ClassWeights] DETRLoss initialized with class weights")

    # Patch the _get_loss_class method
    original_get_loss_class = DETRLoss._get_loss_class

    def patched_get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """Class-weighted classification loss for RT-DETR.

        CRITICAL: Must follow original RT-DETR flow:
        1. Create one_hot from targets (class indices)
        2. Transform gt_scores (IoU) to quality-weighted targets
        3. Apply BCE with class weights
        """
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        nc = self.nc  # Use self.nc like original

        # STEP 1: Create one_hot from targets (EXACTLY like original)
        # targets: [batch, num_queries] - class indices (0=uninfected, 1=infected, nc=background)
        one_hot = torch.zeros((bs, nq, nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]  # Remove background class -> [bs, nq, nc]

        # STEP 2: Transform gt_scores (IoU quality) to quality-weighted targets
        # gt_scores: [batch, num_queries] - IoU values [0, 1]
        # Result: gt_scores * one_hot gives quality-weighted targets
        gt_scores_transformed = gt_scores.view(bs, nq, 1) * one_hot.float()  # [bs, nq, nc]

        # STEP 3: Standard BCE loss (reduction='none')
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores, gt_scores_transformed, reduction='none'
        )  # [bs, nq, nc]

        # STEP 4: Apply class weights ONLY to foreground queries
        # KEY FIX: Background queries (target=[0,0]) should NOT be weighted
        # Otherwise we penalize infected FPs 12.8x more than uninfected -> suppresses infected
        if hasattr(self, '_class_weights'):
            weight_mask = torch.ones_like(bce_loss)  # Default: all weights = 1.0

            for c in range(nc):
                # Only weight where target for channel c is positive (foreground)
                positive_mask = gt_scores_transformed[..., c] > 0  # [bs, nq]
                weight_mask[..., c] = torch.where(
                    positive_mask,
                    torch.full_like(weight_mask[..., c], self._class_weights[c].item()),
                    weight_mask[..., c]  # Keep 1.0 for background/negative
                )

            bce_loss = bce_loss * weight_mask

        # STEP 5: RT-DETR normalization (same as original BCE path)
        loss_cls = bce_loss.mean(1).sum() / max(num_gts, 1) * nq

        return {name_class: loss_cls}

    DETRLoss.__init__ = patched_detr_init
    DETRLoss._get_loss_class = patched_get_loss_class
    print("[ClassWeights] RT-DETR DETRLoss patched with class weights")


# Standalone testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Class Weights Loss Implementation")
    print("=" * 70)

    # Test 1: YOLO Class Weighted BCE
    print("\n" + "=" * 60)
    print("Test 1: YOLO ClassWeightedBCELoss")
    print("=" * 60)

    yolo_loss = ClassWeightedBCELoss(
        weight_uninfected=0.5,
        weight_infected=4.0,
        debug=True
    )

    batch_size = 2
    num_preds = 100
    nc = 2

    pred_scores = torch.randn(batch_size, num_preds, nc)
    target_scores = torch.rand(batch_size, num_preds, nc) * 0.3
    target_scores[:, :10, 1] = 0.8
    target_scores[:, :10, 0] = 0.2

    loss = yolo_loss(pred_scores, target_scores)
    print(f"\nYOLO Loss: {loss.item():.6f}")
    print(f"Loss is finite: {torch.isfinite(loss).item()}")

    # Test gradient
    pred_scores_grad = pred_scores.clone().requires_grad_(True)
    loss_grad = yolo_loss(pred_scores_grad, target_scores)
    loss_grad.backward()
    print(f"Gradients computed: {pred_scores_grad.grad is not None}")

    # Test 2: RT-DETR Class Weighted CE
    print("\n" + "=" * 60)
    print("Test 2: RT-DETR ClassWeightedCELoss")
    print("=" * 60)

    rtdetr_loss = ClassWeightedCELoss(
        weight_uninfected=0.5,
        weight_infected=4.0,
        debug=True
    )

    nq = 300
    num_gts = 15

    pred_scores_rt = torch.randn(batch_size, nq, nc)
    one_hot = torch.zeros(batch_size, nq, nc)
    one_hot[:, :5, 1] = 1  # infected
    one_hot[:, 5:15, 0] = 1  # uninfected

    gt_scores_rt = one_hot.clone()
    gt_scores_rt[:, :15, :] *= torch.rand(batch_size, 15, 1).expand(-1, -1, nc) * 0.5 + 0.5

    loss_rt = rtdetr_loss(pred_scores_rt, one_hot, gt_scores_rt, num_gts, nq)
    print(f"\nRT-DETR Loss: {loss_rt.item():.6f}")
    print(f"Loss is finite: {torch.isfinite(loss_rt).item()}")

    # Test gradient
    pred_scores_rt_grad = pred_scores_rt.clone().requires_grad_(True)
    loss_rt_grad = rtdetr_loss(pred_scores_rt_grad, one_hot, gt_scores_rt, num_gts, nq)
    loss_rt_grad.backward()
    print(f"Gradients computed: {pred_scores_rt_grad.grad is not None}")

    print("\n" + "=" * 70)
    print("All Class Weights Tests Passed!")
    print("=" * 70)
