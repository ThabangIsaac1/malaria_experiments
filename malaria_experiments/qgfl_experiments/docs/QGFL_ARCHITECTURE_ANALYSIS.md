# QGFL Architecture-Specific Integration Analysis

**Date:** 2025-10-05
**Purpose:** Deep analysis of YOLO and RT-DETR loss architectures for QGFL integration

---

## Executive Summary

After thorough investigation of ultralytics source code, here's what we're working with:

| Aspect | YOLO (v8/v11) | RT-DETR |
|--------|---------------|---------|
| **Loss Class** | `v8DetectionLoss` | `RTDETRDetectionLoss` (extends `DETRLoss`) |
| **Classification Loss** | BCE (line 247) | VarifocalLoss OR FocalLoss OR BCE (lines 77-84) |
| **Current Baseline** | Standard BCE, NO focal | VarifocalLoss (use_vfl=True) |
| **Integration Point** | Replace line 247 | Replace lines 77-84 (`_get_loss_class` method) |
| **Architecture** | Anchor-free, Task-Aligned Assigner | DETR-style, Hungarian Matcher |
| **Box Loss** | CIoU + DFL | L1 + GIoU |
| **Key File** | `ultralytics/utils/loss.py` | `ultralytics/models/utils/loss.py` |

---

## Part 1: YOLO Loss Architecture (YOLOv8s, YOLOv11s)

### Current Implementation

**File:** `/ultralytics/utils/loss.py`

```python
class v8DetectionLoss:
    def __init__(self, model, tal_topk=10):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # Line 166
        self.hyp = model.args  # Has box, cls, dfl weights
        self.nc = model.nc  # Number of classes (2 for us)
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)

    def __call__(self, preds, batch):
        # ... target assignment via TaskAlignedAssigner ...

        # Classification loss (LINE 247 - INTEGRATION POINT)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Box loss (KEEP AS-IS)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(...)

        # Apply gains (lines 256-258)
        loss[0] *= self.hyp.box   # 7.5
        loss[1] *= self.hyp.cls   # 0.5
        loss[2] *= self.hyp.dfl   # 1.5

        return loss.sum() * batch_size, loss.detach()
```

### Key Characteristics

1. **Task-Aligned Assigner (TAA):**
   - Assigns targets based on classification score AND IoU quality
   - Creates `target_scores` = weighted combination (alpha=0.5, beta=6.0)
   - Generates ~8400 predictions per image (vs RetinaNet's ~100k anchors)

2. **Classification Loss:**
   - **Current:** Standard BCE
   - **NO focal loss** by default (unlike RetinaNet)
   - **NO class weighting** (treats infected/uninfected equally)
   - Output: scalar loss value summed across all predictions

3. **Loss Components:**
   ```
   Total = (7.5 × Box) + (0.5 × Classification) + (1.5 × DFL)
   ```
   - Box: CIoU (Complete IoU)
   - Classification: BCE (our integration point)
   - DFL: Distribution Focal Loss (for box distribution, NOT classification)

4. **Target Scores:**
   - `target_scores`: Soft labels in [0, 1] (NOT hard 0/1)
   - Computed as: IoU between predicted and GT boxes
   - Used for quality-aware classification training

### QGFL Integration Strategy for YOLO

**Replace line 247 with QGFL:**

```python
# CURRENT (line 247):
loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

# REPLACE WITH:
from qgfl_yolo import QGFLClassificationLoss
self.qgfl_loss = QGFLClassificationLoss(
    nc=self.nc,
    infected_alpha=0.9,
    uninfected_alpha=0.1,
    infected_gamma=8.0,
    uninfected_gamma=4.0,
    difficulty_threshold=0.925,
    quality_margin=0.5,
    quality_factor=2.0,
    uiou_start=2.0,
    uiou_end=0.5
)

loss[1] = self.qgfl_loss(
    pred_scores,           # [batch, num_preds, num_classes] - logits
    target_scores,         # [batch, num_preds, num_classes] - soft targets
    batch["cls"],          # [num_gts] - ground truth class IDs
    target_bboxes,         # [batch, num_preds, 4] - for IoU quality
    pred_bboxes,           # [batch, num_preds, 4] - for IoU quality
    fg_mask,               # [batch, num_preds] - foreground mask
    current_epoch,         # For UIoU decay
    total_epochs           # For UIoU decay
) / target_scores_sum
```

**What QGFL needs access to:**
1. ✓ Predictions (`pred_scores`)
2. ✓ Targets (`target_scores`) - soft labels
3. ✓ Ground truth classes (`batch["cls"]`)
4. ✓ Predicted boxes (`pred_bboxes`) - for quality assessment
5. ✓ Target boxes (`target_bboxes`) - for quality assessment
6. ✓ Foreground mask (`fg_mask`) - which predictions are positive
7. ✓ Current epoch - for UIoU decay
8. ✓ Total epochs - for UIoU decay

---

## Part 2: RT-DETR Loss Architecture

### Current Implementation

**File:** `/ultralytics/models/utils/loss.py`

```python
class DETRLoss(nn.Module):
    def __init__(self, nc, loss_gain=None, aux_loss=True, use_fl=False, use_vfl=False):
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        """INTEGRATION POINT - Lines 77-84"""
        bs, nq = pred_scores.shape[:2]  # batch_size, num_queries (300 for RT-DETR)

        # Create one-hot targets
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]  # Remove background class
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot  # Quality-weighted targets

        # Classification loss (LINES 77-84 - INTEGRATION POINT)
        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)  # ← RT-DETR uses THIS
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()

        return {f'loss_class{postfix}': loss_cls.squeeze() * self.loss_gain['class']}

class RTDETRDetectionLoss(DETRLoss):
    """Extends DETRLoss with denoising training"""
    # Initialization: RTDETRDetectionModel calls this with use_vfl=True
    # So RT-DETR baseline uses VarifocalLoss, NOT standard BCE!
```

### Key Characteristics

1. **Hungarian Matcher:**
   - Bipartite matching between 300 queries and N ground truths
   - Cost function: weighted sum of classification + bbox + giou costs
   - Creates 1:1 assignment (each GT matched to at most one query)

2. **Classification Loss:**
   - **Current:** **VarifocalLoss** (NOT BCE!)
   - VarifocalLoss formula:
     ```python
     weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
     loss = BCE(pred, gt) * weight
     ```
   - Uses quality-weighted targets (`gt_scores` = IoU-based)
   - Already has focal mechanism (alpha=0.75, gamma=2.0)

3. **Architecture Differences from YOLO:**
   - 300 queries (vs YOLO's 8400 predictions)
   - No anchor boxes - pure set prediction
   - Auxiliary losses from intermediate decoder layers
   - Denoising training (dn_loss) for better convergence

4. **Loss Components:**
   ```
   Total = (1.0 × Class) + (5.0 × BBox) + (2.0 × GIoU)
   + Auxiliary losses (if aux_loss=True)
   + Denoising losses (if dn_meta provided)
   ```

### QGFL Integration Strategy for RT-DETR

**Replace lines 77-84 in `_get_loss_class` method:**

```python
# CURRENT (lines 77-84):
if self.fl:
    if num_gts and self.vfl:
        loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
    else:
        loss_cls = self.fl(pred_scores, one_hot.float())
    loss_cls /= max(num_gts, 1) / nq
else:
    loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()

# REPLACE WITH:
from qgfl_rtdetr import QGFLClassificationLoss
if not hasattr(self, 'qgfl_loss'):
    self.qgfl_loss = QGFLClassificationLoss(
        nc=self.nc,
        infected_alpha=0.9,
        uninfected_alpha=0.1,
        infected_gamma=8.0,
        uninfected_gamma=4.0,
        difficulty_threshold=0.925,
        quality_margin=0.5,
        quality_factor=2.0,
        uiou_start=2.0,
        uiou_end=0.5
    )

loss_cls = self.qgfl_loss(
    pred_scores,           # [batch, 300, num_classes] - logits
    one_hot,               # [batch, 300, num_classes] - hard targets
    gt_scores,             # [batch, 300] - quality scores (IoU-based)
    targets,               # [batch, 300] - class indices
    num_gts,               # Number of ground truths
    nq,                    # Number of queries (300)
    current_epoch,         # For UIoU decay
    total_epochs           # For UIoU decay
)
```

**What QGFL needs access to:**
1. ✓ Predictions (`pred_scores`)
2. ✓ One-hot targets (`one_hot`)
3. ✓ Quality scores (`gt_scores`) - IoU-based
4. ✓ Class targets (`targets`)
5. ✓ Number of GTs (`num_gts`)
6. ✓ Number of queries (`nq`)
7. ✓ Current epoch - for UIoU decay
8. ✓ Total epochs - for UIoU decay

---

## Part 3: Critical Differences & Implications

| Aspect | YOLO | RT-DETR | Implication for QGFL |
|--------|------|---------|---------------------|
| **Number of predictions** | ~8400 | 300 | YOLO has more easy negatives → higher γ might be OK |
| **Assignment strategy** | Task-Aligned (soft) | Hungarian (hard 1:1) | YOLO uses soft targets, RT-DETR uses one-hot |
| **Current focal mechanism** | NONE | VarifocalLoss (α=0.75, γ=2.0) | RT-DETR already handles easy examples |
| **Target quality** | IoU-based soft labels | IoU-weighted one-hot | Both use quality, different formulations |
| **Class imbalance handling** | NONE | NONE | Both need QGFL equally |

### Key Insights

1. **RT-DETR already uses focal loss!**
   - Baseline isn't "standard BCE" - it's VarifocalLoss
   - QGFL will be **replacing** focal loss, not adding it
   - Need to show QGFL's class-specific + quality-guided approach > generic varifocal

2. **Different target formats:**
   - YOLO: Soft targets `target_scores` ∈ [0, 1] (IoU-quality weighted)
   - RT-DETR: Hard one-hot + separate quality scores
   - QGFL implementation must handle both

3. **Gamma values might need adjustment:**
   - Paper used γ=8.0/4.0 for RetinaNet (~100k anchors)
   - YOLO has ~8400 predictions → might be OK
   - RT-DETR has only 300 queries → γ=8.0 might be **too aggressive**
   - **Recommendation:** Start with paper's values, monitor stability

4. **UIoU component might conflict with YOLO's DFL:**
   - YOLO already has Distribution Focal Loss for box quality
   - UIoU might be redundant
   - **Recommendation:** Test with/without UIoU for YOLO (ablation)

---

## Part 4: Implementation Plan

### Phase 1: Create Separate Loss Modules

#### File 1: `src/losses/qgfl_core.py`
**Purpose:** Core QGFL components (shared between YOLO and RT-DETR)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QGFLCore(nn.Module):
    """Core QGFL loss computation - architecture agnostic"""
    def __init__(self, infected_alpha, uninfected_alpha, infected_gamma,
                 uninfected_gamma, difficulty_threshold, quality_margin,
                 quality_factor, uiou_start, uiou_end):
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

    def compute_gamma_eff(self, pt, is_infected):
        """
        Compute effective gamma with class-specific + difficulty-aware scaling

        Args:
            pt: Predicted probability for true class [N]
            is_infected: Boolean mask [N]

        Returns:
            gamma_eff: Effective gamma values [N]
        """
        # Class-specific base gamma
        base_gamma = torch.where(
            is_infected,
            torch.tensor(2.0, device=pt.device),  # Standard focal loss gamma
            torch.tensor(2.0, device=pt.device)
        )

        # Class-specific max gamma
        max_gamma = torch.where(
            is_infected,
            torch.tensor(self.infected_gamma, device=pt.device),
            torch.tensor(self.uninfected_gamma, device=pt.device)
        )

        # Difficulty-aware scaling with thresholding
        raw_difficulty = 1.0 - pt
        difficulty = torch.clamp(
            (raw_difficulty - self.difficulty_threshold) / (1.0 - self.difficulty_threshold),
            min=0.0
        )

        # Interpolate between base and max gamma
        gamma_eff = base_gamma + (max_gamma - base_gamma) * difficulty

        return gamma_eff

    def compute_quality_weight(self, predictions, targets):
        """
        Compute quality-guided weighting

        Args:
            predictions: Predicted class probabilities [N, C]
            targets: Target class probabilities [N, C]

        Returns:
            quality_weight: Weighting factor [N]
        """
        # Quality = absolute difference between prediction and target
        quality = torch.abs(predictions - targets).sum(dim=-1)

        # Adjust quality
        quality_adjusted = torch.clamp(quality - self.quality_margin, min=0.0)

        # Apply exponential scaling
        quality_weight = torch.clamp(
            quality_adjusted ** self.quality_factor,
            max=10.0
        )

        return quality_weight

    def compute_uiou_ratio(self, current_epoch, total_epochs):
        """
        Compute UIoU ratio with linear decay

        Args:
            current_epoch: Current training epoch
            total_epochs: Total number of epochs

        Returns:
            uiou_ratio: Current UIoU weighting
        """
        progress = current_epoch / max(total_epochs, 1)
        uiou_ratio = self.uiou_start + (self.uiou_end - self.uiou_start) * progress
        return uiou_ratio
```

#### File 2: `src/losses/qgfl_yolo.py`
**Purpose:** YOLO-specific QGFL implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .qgfl_core import QGFLCore
from ultralytics.utils.metrics import bbox_iou

class QGFLYOLOLoss(QGFLCore):
    """QGFL loss for YOLO (v8/v11) architecture"""

    def __init__(self, nc, **qgfl_params):
        super().__init__(**qgfl_params)
        self.nc = nc

    def forward(self, pred_scores, target_scores, gt_classes,
                pred_bboxes, target_bboxes, fg_mask,
                current_epoch=0, total_epochs=200):
        """
        Compute QGFL loss for YOLO

        Args:
            pred_scores: [batch, num_preds, nc] - predicted logits
            target_scores: [batch, num_preds, nc] - soft targets (IoU-weighted)
            gt_classes: [num_gts] - ground truth class IDs
            pred_bboxes: [batch, num_preds, 4] - predicted boxes
            target_bboxes: [batch, num_preds, 4] - target boxes
            fg_mask: [batch, num_preds] - foreground mask
            current_epoch: Current epoch
            total_epochs: Total epochs

        Returns:
            loss: Scalar QGFL loss
        """
        # Get probabilities
        pred_prob = pred_scores.sigmoid()

        # Compute pt for focal loss (YOLO uses soft targets)
        pt = pred_prob * target_scores + (1 - pred_prob) * (1 - target_scores)

        # Determine which predictions are infected class (class 1 in binary)
        is_infected = target_scores[..., 1] > target_scores[..., 0]

        # Flatten for easier computation
        pt_flat = pt[fg_mask].view(-1)
        pred_flat = pred_prob[fg_mask].view(-1, self.nc)
        target_flat = target_scores[fg_mask].view(-1, self.nc)
        is_infected_flat = is_infected[fg_mask].view(-1)

        # Compute class-specific alpha
        alpha = torch.where(
            is_infected_flat.unsqueeze(-1),
            torch.tensor(self.infected_alpha, device=pt.device),
            torch.tensor(self.uninfected_alpha, device=pt.device)
        ).expand_as(pred_flat)

        # Compute effective gamma (difficulty + class aware)
        gamma_eff = self.compute_gamma_eff(pt_flat, is_infected_flat.unsqueeze(-1).expand_as(pred_flat))

        # Compute quality weight
        quality_weight = self.compute_quality_weight(pred_flat, target_flat)

        # Compute UIoU ratio
        uiou_ratio = self.compute_uiou_ratio(current_epoch, total_epochs)

        # Standard focal loss with modulating factor
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores[fg_mask].view(-1, self.nc),
            target_scores[fg_mask].view(-1, self.nc),
            reduction='none'
        )

        # Modulating factor: (1 - pt)^gamma_eff
        modulating_factor = (1.0 - pt_flat.view(-1, self.nc)) ** gamma_eff

        # Apply all components
        loss = alpha * modulating_factor * (1.0 + quality_weight.unsqueeze(-1)) * uiou_ratio * bce_loss

        return loss.sum()
```

#### File 3: `src/losses/qgfl_rtdetr.py`
**Purpose:** RT-DETR-specific QGFL implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .qgfl_core import QGFLCore

class QGFLRTDETRLoss(QGFLCore):
    """QGFL loss for RT-DETR architecture"""

    def __init__(self, nc, **qgfl_params):
        super().__init__(**qgfl_params)
        self.nc = nc

    def forward(self, pred_scores, one_hot, gt_scores, targets,
                num_gts, nq, current_epoch=0, total_epochs=200):
        """
        Compute QGFL loss for RT-DETR

        Args:
            pred_scores: [batch, 300, nc] - predicted logits
            one_hot: [batch, 300, nc] - one-hot targets
            gt_scores: [batch, 300, nc] - IoU-quality weighted targets
            targets: [batch, 300] - class indices
            num_gts: Number of ground truths
            nq: Number of queries (300)
            current_epoch: Current epoch
            total_epochs: Total epochs

        Returns:
            loss: Scalar QGFL loss
        """
        bs = pred_scores.shape[0]

        # Get probabilities
        pred_prob = pred_scores.sigmoid()

        # Compute pt (RT-DETR uses quality-weighted one-hot)
        pt = pred_prob * gt_scores + (1 - pred_prob) * (1 - gt_scores)

        # Determine infected class (class 1 in binary)
        is_infected = (targets == 1).unsqueeze(-1).expand(-1, -1, self.nc)

        # Compute class-specific alpha
        alpha = torch.where(
            is_infected,
            torch.tensor(self.infected_alpha, device=pt.device),
            torch.tensor(self.uninfected_alpha, device=pt.device)
        )

        # Compute effective gamma
        gamma_eff = self.compute_gamma_eff(pt.view(-1), is_infected.view(-1)).view_as(pt)

        # Compute quality weight
        quality_weight = self.compute_quality_weight(pred_prob, gt_scores)

        # Compute UIoU ratio
        uiou_ratio = self.compute_uiou_ratio(current_epoch, total_epochs)

        # Standard BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_scores,
            gt_scores,
            reduction='none'
        )

        # Modulating factor
        modulating_factor = (1.0 - pt) ** gamma_eff

        # Apply all components
        loss = alpha * modulating_factor * (1.0 + quality_weight.unsqueeze(-1)) * uiou_ratio * bce_loss

        # Normalize by number of GTs (same as VarifocalLoss)
        loss = loss.mean(1).sum() / max(num_gts, 1) * nq

        return loss
```

### Phase 2: Integration into Training Pipeline

#### Modify `cluster_run_baseline.py`:

```python
# Add argument for loss type
parser.add_argument('--loss-type', type=str, default='baseline',
                   choices=['baseline', 'qgfl'],
                   help='Loss function type')

# In training setup section:
if config.loss_type == 'qgfl':
    # Override model loss function
    if config.model_name in ['yolov8s', 'yolov11s', 'yolo11s']:
        from src.losses.qgfl_yolo import QGFLYOLOLoss
        # Patch YOLO loss (implementation details in Phase 3)
    elif config.model_name == 'rtdetr':
        from src.losses.qgfl_rtdetr import QGFLRTDETRLoss
        # Patch RT-DETR loss (implementation details in Phase 3)
```

### Phase 3: Monkey-Patching Strategy

Since we can't easily modify ultralytics source directly, we'll use monkey-patching:

```python
def patch_yolo_with_qgfl(model, qgfl_params):
    """Replace YOLO's BCE classification loss with QGFL"""
    original_loss_class = model.model.loss.__call__
    qgfl_loss = QGFLYOLOLoss(nc=model.model.nc, **qgfl_params)

    def qgfl_forward(self, preds, batch):
        # Call original to get target assignment
        loss_original, loss_items = original_loss_class(self, preds, batch)

        # Replace classification loss component
        # [Implementation details...]

    model.model.loss.__call__ = qgfl_forward.__get__(model.model.loss, type(model.model.loss))

def patch_rtdetr_with_qgfl(model, qgfl_params):
    """Replace RT-DETR's VarifocalLoss with QGFL"""
    original_get_loss_class = model.model.loss._get_loss_class
    qgfl_loss = QGFLRTDETRLoss(nc=model.model.nc, **qgfl_params)

    def qgfl_get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
        # Replace with QGFL
        # [Implementation details...]

    model.model.loss._get_loss_class = qgfl_get_loss_class.__get__(model.model.loss, type(model.model.loss))
```

---

## Part 5: Testing Strategy

### Stability Test (5 epochs, batch=4)

```bash
# YOLO stability test
python cluster_run_baseline.py --model yolov8s --dataset d1 --task binary \
    --epochs 5 --batch-size 4 --loss-type qgfl

# RT-DETR stability test
python cluster_run_baseline.py --model rtdetr-l --dataset d1 --task binary \
    --epochs 5 --batch-size 4 --loss-type qgfl
```

**What to monitor:**
1. Loss doesn't collapse to zero
2. Loss doesn't explode (NaN/Inf)
3. Gradients are flowing (watch W&B gradient charts)
4. Training completes without errors
5. Validation metrics are reasonable (not random)

**Red flags:**
- Loss < 0.001 after epoch 1 → Gamma too high
- Loss > 100 after epoch 1 → Something wrong with normalization
- NaN/Inf → Numerical stability issue
- mAP < 0.1 → Model not learning

### Full Experiment (200 epochs, batch=16)

Only proceed if stability tests pass for both architectures.

---

## Part 6: Expected Challenges & Solutions

### Challenge 1: Gamma too aggressive for RT-DETR

**Symptom:** Loss collapses, model stops learning
**Solution:** Reduce gamma: infected=6.0, uninfected=3.0

### Challenge 2: UIoU conflicts with YOLO's DFL

**Symptom:** Performance worse than baseline
**Solution:** Disable UIoU for YOLO (set start=1.0, end=1.0)

### Challenge 3: Soft targets (YOLO) vs Hard targets (RT-DETR)

**Symptom:** Inconsistent behavior between architectures
**Solution:** Already handled - separate implementations

### Challenge 4: Different normalization schemes

**Symptom:** Loss magnitudes very different between models
**Solution:** Each implementation uses architecture-specific normalization

---

## Part 7: Success Criteria

### Minimum Viable Success (Stability Test)

- ✓ Training completes without errors
- ✓ Loss converges (not zero, not exploding)
- ✓ Validation mAP > 0.3 (shows model is learning)
- ✓ Per-class mAP logged correctly to W&B

### Full Success (200 epochs)

**Compared to baseline:**
- ↑ Infected class recall (especially at 1-3% density)
- ↑ Infected class F1 score
- ↑ Per-class mAP@50-95 for infected
- ≈ Uninfected class metrics (maintain, don't degrade)
- ↑ Overall mAP@50-95

**Target improvements (from paper):**
- D1: +46% recall @ 1-3% density
- D2: +93% recall @ 1-3% density
- D3: +8% recall @ 1-3% density

---

## Part 8: File Structure

```
malaria_experiments/qgfl_experiments/
├── src/
│   └── losses/
│       ├── __init__.py
│       ├── qgfl_core.py          # Shared QGFL components
│       ├── qgfl_yolo.py          # YOLO-specific implementation
│       └── qgfl_rtdetr.py        # RT-DETR-specific implementation
├── cluster_run_baseline.py       # Modified to support --loss-type qgfl
├── configs/
│   └── qgfl_config.py            # QGFL hyperparameters
└── docs/
    ├── QGFL_ARCHITECTURE_ANALYSIS.md  # This document
    └── QGFL_INTEGRATION_PLAN.md       # Implementation checklist
```

---

## Conclusion

**We have a clear path forward:**

1. ✓ **Understand architectures** - COMPLETE (this document)
2. **Create loss modules** - src/losses/qgfl_*.py (3 files)
3. **Integrate via monkey-patching** - Modify cluster_run_baseline.py
4. **Stability test** - 5 epochs, batch=4, both architectures
5. **Full experiments** - 200 epochs, batch=16, 18 total runs

**Key insights:**
- YOLO uses BCE (needs QGFL)
- RT-DETR uses VarifocalLoss (QGFL must beat this, not BCE)
- Separate implementations required (different target formats)
- Monkey-patching avoids modifying ultralytics source
- Stability testing critical before full runs

**Next action:** Implement `src/losses/qgfl_core.py`
