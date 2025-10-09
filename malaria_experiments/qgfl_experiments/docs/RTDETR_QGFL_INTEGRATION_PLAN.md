# RT-DETR QGFL Integration Plan

**Author:** Thabang Isaka
**Date:** 2025-10-05
**Status:** Ready for Implementation

## Executive Summary

This document provides a detailed implementation plan for integrating QGFL (Quality-Guided Focal Loss) into RT-DETR architecture. The integration is more complex than YOLO due to RT-DETR's Hungarian matcher, query-based predictions, and VarifocalLoss baseline (not simple BCE).

**Key Challenge:** RT-DETR baseline already uses VarifocalLoss (α=0.75, γ=2.0), so QGFL must demonstrate clear advantages over an already sophisticated loss function.

---

## 1. Architecture Comparison: YOLO vs RT-DETR

### YOLO Architecture (Successfully Integrated ✓)
```
Predictions: ~8400 per image (anchor-based dense predictions)
Targets: Soft targets (IoU-weighted, ∈ [0, 1])
Assignment: Task-Aligned Assigner (soft matching)
Loss: BCE (simple binary cross-entropy)
Integration: Simple replacement of BCE with QGFL
```

### RT-DETR Architecture (To Be Integrated)
```
Predictions: 300 queries per image (sparse, learned)
Targets: One-hot + quality scores (separate components)
Assignment: Hungarian Matcher (bipartite hard matching)
Loss: VarifocalLoss (α=0.75, γ=2.0) - already sophisticated!
Integration: Replace VarifocalLoss with QGFL
```

**Critical Insight:** RT-DETR's VarifocalLoss is already class-imbalance aware (α=0.75 favors positives). QGFL must add value through:
1. **Difficulty-aware gamma scaling** (Level 2)
2. **Class-specific gamma** (Level 3: infected γ=8.0, uninfected γ=4.0)
3. **Difficulty thresholding** (Level 4: pt > 0.925)
4. **Quality-guided weighting** (Level 5)
5. **UIoU decay** (Level 5)

---

## 2. Current RT-DETR Loss Architecture

### File Location
```
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/ultralytics/models/utils/loss.py
```

### Key Classes
1. **DETRLoss** (base class, lines 13-288)
2. **RTDETRDetectionLoss** (extends DETRLoss, lines 291-357)
3. **HungarianMatcher** (in ops.py, lines 12-117)

### Current Classification Loss Implementation
**Location:** `DETRLoss._get_loss_class` (lines 66-86)

```python
def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
    """Computes classification loss based on predictions and ground truth."""
    bs, nq = pred_scores.shape[:2]  # batch_size, num_queries (300)

    # Create one-hot targets
    one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
    one_hot.scatter_(2, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]  # Remove background class

    # Apply quality scores (IoU-weighted)
    gt_scores = gt_scores.view(bs, nq, 1) * one_hot

    # CURRENT LOSS SELECTION (lines 77-84):
    if self.fl:
        if num_gts and self.vfl:
            # RT-DETR uses this branch: VarifocalLoss
            loss_cls = self.vfl(pred_scores, gt_scores, one_hot)  # ← TARGET FOR QGFL
        else:
            # Fallback: FocalLoss
            loss_cls = self.fl(pred_scores, one_hot.float())
        loss_cls /= max(num_gts, 1) / nq  # Normalize by number of GTs
    else:
        # Fallback: BCE
        loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()

    return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}
```

### VarifocalLoss Parameters
**Location:** `ultralytics/utils/loss.py` (VarifocalLoss class)

```python
VarifocalLoss(
    alpha=0.75,  # Favors positive (minority) class
    gamma=2.0,   # Difficulty modulation (fixed)
    iou_weighted=True  # Uses IoU quality scores
)
```

**Key Observation:** VarifocalLoss already has:
- ✓ Class imbalance handling (α=0.75)
- ✓ Difficulty modulation (γ=2.0, but fixed)
- ✓ Quality weighting (IoU-based)

**What QGFL Adds:**
- ✗ Difficulty-aware gamma (VarifocalLoss has fixed γ=2.0)
- ✗ Class-specific gamma (infected γ=8.0 vs uninfected γ=4.0)
- ✗ Difficulty thresholding (pt > 0.925 check)
- ✗ Enhanced quality-guided weighting
- ✗ UIoU decay over training

---

## 3. Integration Strategy

### Option A: Monkey-Patching (Recommended, Same as YOLO)
**Pros:**
- ✓ No ultralytics source code modification
- ✓ Pickle-safe (YOLO validated this)
- ✓ Easy deployment to cluster
- ✓ Follows proven YOLO integration approach

**Cons:**
- ✗ Requires understanding of exact method signature
- ✗ Must handle one-hot + quality scores correctly

**Implementation:**
```python
# In cluster_run_qgfl.py (or new cluster_run_rtdetr_qgfl.py)

# 1. Import RT-DETR loss module
import ultralytics.models.utils.loss as rtdetr_loss_module

# 2. Initialize QGFL RT-DETR loss
qgfl_rtdetr = QGFLRTDETRLoss(
    nc=2,
    infected_alpha=0.9,
    uninfected_alpha=0.1,
    infected_gamma=8.0,
    uninfected_gamma=4.0,
    difficulty_threshold=0.925,
    quality_margin=0.5,
    quality_factor=2.0,
    uiou_start=2.0,
    uiou_end=0.5,
    debug=args.qgfl_debug
)

# 3. Store references for closure (pickle-safe)
_qgfl_rtdetr_loss = qgfl_rtdetr
_qgfl_args_epochs = args.epochs

# 4. Create replacement method for _get_loss_class
def qgfl_get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=""):
    """QGFL-enhanced classification loss for RT-DETR"""
    name_class = f"loss_class{postfix}"
    bs, nq = pred_scores.shape[:2]

    # Create one-hot targets (same as original)
    one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
    one_hot.scatter_(2, targets.unsqueeze(-1), 1)
    one_hot = one_hot[..., :-1]

    # Apply quality scores (same as original)
    gt_scores_weighted = gt_scores.view(bs, nq, 1) * one_hot

    # *** QGFL REPLACEMENT ***
    if num_gts:
        loss_cls = _qgfl_rtdetr_loss(
            pred_scores=pred_scores,
            one_hot=one_hot,
            gt_scores=gt_scores_weighted,
            num_gts=num_gts,
            nq=nq,
            current_epoch=getattr(self, '_current_epoch', 0),
            total_epochs=_qgfl_args_epochs
        )
    else:
        # No ground truths, return zero loss
        loss_cls = torch.tensor(0.0, device=pred_scores.device)

    return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

# 5. Monkey-patch the method
rtdetr_loss_module.DETRLoss._get_loss_class = qgfl_get_loss_class

# 6. Track epoch for UIoU decay (same as YOLO)
def qgfl_on_train_epoch_start(trainer):
    """Update current epoch in loss function"""
    if hasattr(trainer, 'loss') and hasattr(trainer.loss, '_current_epoch'):
        trainer.loss._current_epoch = trainer.epoch

trainer.add_callback("on_train_epoch_start", qgfl_on_train_epoch_start)
```

### Option B: Subclass DETRLoss (Not Recommended)
**Pros:**
- ✓ Clean OOP design
- ✓ Full control over loss computation

**Cons:**
- ✗ Pickle issues (YOLO had these)
- ✗ Harder to maintain ultralytics compatibility
- ✗ More complex deployment

**Decision:** Use Option A (Monkey-Patching)

---

## 4. Detailed Implementation Steps

### Step 1: Verify Standalone QGFL RT-DETR Code
**File:** `malaria_experiments/qgfl_experiments/src/losses/qgfl_rtdetr.py`

**Status:** ✓ Already implemented (239 lines)

**Test:** Run standalone test
```bash
cd malaria_experiments/qgfl_experiments/src/losses
python qgfl_rtdetr.py
```

**Expected Output:**
```
Testing QGFL RT-DETR Loss...
============================================================
Test: RT-DETR Forward Pass
============================================================
[DEBUG] RT-DETR Forward:
  Batch size: 2
  Num queries: 300
  Num classes: 2
  ...
Output:
  Loss: X.XXXXXX
  Loss is finite: True
  Loss > 0: True
============================================================
Test: Gradient Flow
============================================================
Gradients computed: True
...
QGFL RT-DETR Tests Passed! ✓
```

### Step 2: Create RT-DETR Training Script
**New File:** `malaria_experiments/qgfl_experiments/cluster_run_rtdetr_qgfl.py`

**Approach:**
1. Copy `cluster_run_qgfl.py` as template
2. Modify model selection to RT-DETR models:
   ```python
   RTDETR_MODELS = {
       'rtdetr-s': 'rtdetr-l.pt',  # Small
       'rtdetr-m': 'rtdetr-x.pt',  # Medium
       'rtdetr-l': 'rtdetr-l.pt',  # Large
   }
   ```
3. Update monkey-patching section for RT-DETR (see Option A above)
4. Update experiment naming:
   ```python
   experiment_name = f"laptop_rtdetr-l_d1_binary_qgfl"
   ```
5. Keep same dataset configs (D1, D2, D3)

### Step 3: Run 5-Epoch Smoke Test (CPU)
**Command:**
```bash
cd malaria_experiments/qgfl_experiments
python cluster_run_rtdetr_qgfl.py \
    --model rtdetr-l \
    --dataset d1 \
    --loss-type qgfl \
    --epochs 5 \
    --batch-size 8 \
    --device cpu \
    --qgfl-debug
```

**Expected Behavior:**
```
[QGFL] Monkey-patching RT-DETR DETRLoss._get_loss_class...
[QGFL] ✓ RT-DETR DETRLoss._get_loss_class patched successfully
[QGFL] RT-DETR loss integration: ACTIVE

Training starts...
Epoch 1/5: [QGFL-RTDETR] Computing loss...
  pt range: [0.XXXX, 0.XXXX]
  alpha range: [0.1000, 0.9000]
  gamma_eff range: [4.0000, 8.0000]
  ...
```

**Compare to Baseline (VarifocalLoss):**
```bash
python cluster_run_baseline.py \
    --model rtdetr-l \
    --dataset d1 \
    --epochs 5 \
    --batch-size 8 \
    --device cpu
```

### Step 4: Analyze Results
**Metrics to Compare:**

| Metric | Baseline (VarifocalLoss) | QGFL | Improvement |
|--------|--------------------------|------|-------------|
| Infected Recall (Test) | ? | ? | ? |
| Low Parasitemia (0-1%) Recall | ? | ? | ? |
| Overall mAP@50 | ? | ? | ? |
| Val-Test Gap | ? | ? | ? |

**Success Criteria:**
- ✓ QGFL infected recall > VarifocalLoss infected recall
- ✓ QGFL low parasitemia recall > VarifocalLoss (ideally 100%)
- ✓ No overfitting (val-test gap < 10%)
- ✓ Training stable (no NaN losses)

### Step 5: Deploy to Cluster (if smoke test succeeds)
**Experiments:** 6 total
- RT-DETR-L + D1 (QGFL) [200 epochs]
- RT-DETR-L + D2 (QGFL) [200 epochs]
- RT-DETR-L + D3 (QGFL) [200 epochs]
- RT-DETR-L + D1 (Baseline) [200 epochs]
- RT-DETR-L + D2 (Baseline) [200 epochs]
- RT-DETR-L + D3 (Baseline) [200 epochs]

**Estimated Time:** ~6 hours each × 6 = 36 hours total

---

## 5. Technical Challenges and Solutions

### Challenge 1: One-Hot vs Soft Targets
**Issue:** RT-DETR uses one-hot targets + separate quality scores, YOLO uses soft targets (IoU-weighted)

**Solution:**
```python
# YOLO approach (single soft target):
target_scores = [0.0, 0.85]  # IoU=0.85 for infected class

# RT-DETR approach (separate components):
one_hot = [0, 1]  # Hard class label
gt_scores = 0.85  # IoU quality (applied separately)
gt_scores_weighted = gt_scores * one_hot = [0.0, 0.85]  # Same result!
```

**Status:** ✓ Already handled in qgfl_rtdetr.py (lines 100-107)

### Challenge 2: Hungarian Matcher Integration
**Issue:** QGFL doesn't need to modify Hungarian matcher, but must work with its outputs

**Solution:**
- Hungarian matcher runs BEFORE loss computation (line 231 in DETRLoss._get_loss)
- Matcher outputs: `match_indices = [(query_ids, gt_ids), ...]` per batch
- Loss function receives: `targets` (matched class IDs) and `gt_scores` (matched IoU scores)
- QGFL only replaces classification loss, not matching process

**Status:** ✓ No changes needed to matcher, QGFL works with matched predictions

### Challenge 3: Epoch Tracking for UIoU Decay
**Issue:** RT-DETR loss function doesn't inherently track current epoch

**Solution:**
```python
# In training script (same as YOLO):
def qgfl_on_train_epoch_start(trainer):
    """Update current epoch in loss function for UIoU decay"""
    if hasattr(trainer, 'loss'):
        trainer.loss._current_epoch = trainer.epoch

trainer.add_callback("on_train_epoch_start", qgfl_on_train_epoch_start)
```

**Status:** ✓ Implemented in cluster_run_qgfl.py (lines 671-681)

### Challenge 4: Beating VarifocalLoss Baseline
**Issue:** RT-DETR baseline (VarifocalLoss) is already sophisticated

**QGFL Advantages:**
1. **Difficulty-aware gamma:** VarifocalLoss uses fixed γ=2.0, QGFL uses dynamic γ ∈ [4.0, 8.0] based on pt
2. **Class-specific gamma:** VarifocalLoss same γ for all classes, QGFL has infected γ=8.0 vs uninfected γ=4.0
3. **Difficulty thresholding:** QGFL checks pt > 0.925 for gamma scaling
4. **Enhanced quality weighting:** QGFL adds explicit quality margin and factor
5. **UIoU decay:** Progressive focus on harder examples over training

**Expected Outcome:** QGFL should show 10-20% relative improvement in infected recall (based on YOLO results)

---

## 6. Success Metrics

### Validation Phase (5-epoch smoke test)
- [ ] Training completes without errors
- [ ] No NaN/inf losses
- [ ] QGFL infected recall > 50% (baseline was 7% for YOLO)
- [ ] Loss curves show smooth convergence

### Full Training Phase (200 epochs)
- [ ] QGFL infected recall > VarifocalLoss infected recall
- [ ] QGFL low parasitemia (0-1%) recall ≥ 80% (ideally 100%)
- [ ] QGFL overall mAP@50 competitive with VarifocalLoss (±5%)
- [ ] No overfitting (val-test gap < 10%)

### Clinical Impact
- [ ] Early detection capability (0-1% parasitemia) significantly improved
- [ ] Total infected cells detected > baseline
- [ ] Consistent performance across all parasitemia levels

---

## 7. Implementation Checklist

### Phase 1: Preparation (COMPLETED ✓)
- [x] Delete old results/runs/wandb folders
- [x] Analyze RT-DETR loss architecture
- [x] Review Hungarian matcher implementation
- [x] Create integration plan document
- [x] Verify standalone qgfl_rtdetr.py tests pass

### Phase 2: Integration (COMPLETED ✓)
- [x] ~~Create cluster_run_rtdetr_qgfl.py script~~ UNIFIED: Use cluster_run_qgfl.py with --model rtdetr
- [x] Implement monkey-patching for DETRLoss._get_loss_class (lines 216-273)
- [x] Add epoch tracking callback (lines 671-681)
- [x] Test import and initialization (standalone test passed)

### Phase 3: Smoke Testing
- [ ] Run 5-epoch QGFL smoke test (CPU)
- [ ] Run 5-epoch baseline smoke test (CPU)
- [ ] Compare results (infected recall, low parasitemia)
- [ ] Verify no overfitting
- [ ] Document findings

### Phase 4: Cluster Deployment (if smoke test succeeds)
- [ ] Transfer scripts to cluster
- [ ] Run 6 experiments (3 datasets × 2 loss types)
- [ ] Monitor training progress
- [ ] Collect results

### Phase 5: Analysis
- [ ] Extract per-class metrics
- [ ] Prevalence-stratified analysis
- [ ] Generalization analysis (val-test gap)
- [ ] Clinical impact assessment
- [ ] Update documentation

---

## 8. File Structure

```
malaria_experiments/qgfl_experiments/
├── src/losses/
│   ├── qgfl_core.py              # Shared QGFL components (381 lines) ✓
│   ├── qgfl_yolo.py              # YOLO-specific QGFL (211 lines) ✓
│   └── qgfl_rtdetr.py            # RT-DETR-specific QGFL (239 lines) ✓
├── cluster_run_qgfl.py           # UNIFIED: YOLO + RT-DETR QGFL training ✓
│                                 # Auto-detects model type and applies correct loss
├── cluster_run_baseline.py       # YOLO baseline training ✓
└── docs/
    ├── QGFL_IMPLEMENTATION_SUMMARY.md  # YOLO results (558 lines) ✓
    └── RTDETR_QGFL_INTEGRATION_PLAN.md # This document ✓
```

---

## 9. Next Steps

**COMPLETED:**
1. ✓ Verify standalone qgfl_rtdetr.py tests pass
2. ✓ ~~Create cluster_run_rtdetr_qgfl.py~~ Unified into cluster_run_qgfl.py
3. ✓ Implement RT-DETR monkey-patching (lines 216-273)
4. ✓ Add epoch tracking callback (lines 671-681)

**Ready for Testing (Next):**
5. Run 5-epoch QGFL smoke test: `--model rtdetr --dataset d1 --loss-type qgfl --epochs 5`
6. Run 5-epoch baseline smoke test: `--model rtdetr --dataset d1 --loss-type baseline --epochs 5`
7. Compare infected recall and low parasitemia performance
8. Document results and compare to YOLO QGFL

**Mid-term (This Week):**
9. If smoke tests succeed, deploy to cluster
10. Run full 200-epoch experiments (6 total)
11. Analyze results and compare to YOLO QGFL
12. Write final report

---

## 10. Risk Assessment

### High Risk
- **VarifocalLoss baseline is strong:** QGFL must demonstrate clear advantages
- **Hungarian matcher complexity:** Any bugs hard to debug

### Medium Risk
- **RT-DETR slower training:** May need more batch tuning for cluster
- **Different hyperparameters:** RT-DETR may need different γ values than YOLO

### Low Risk
- **Pickle issues:** Monkey-patching validated on YOLO
- **Epoch tracking:** Same callback approach as YOLO

### Mitigation
- Start with 5-epoch smoke tests (low cost, fast validation)
- Use same QGFL parameters as YOLO (γ=8.0/4.0) initially
- Monitor loss curves closely for instability
- Keep baseline experiments running in parallel for comparison

---

## 11. References

### Code Files
- RT-DETR Loss: `/Library/.../ultralytics/models/utils/loss.py` (lines 291-357)
- Hungarian Matcher: `/Library/.../ultralytics/models/utils/ops.py` (lines 12-117)
- VarifocalLoss: `/Library/.../ultralytics/utils/loss.py`
- QGFL RT-DETR: `src/losses/qgfl_rtdetr.py`

### Documentation
- QGFL YOLO Results: `docs/QGFL_IMPLEMENTATION_SUMMARY.md`
- QGFL Architecture: `docs/QGFL_ARCHITECTURE_ANALYSIS.md`
- QGFL Quick Reference: `docs/QGFL_QUICK_REFERENCE.md`

### Papers
- RT-DETR: "DETRs Beat YOLOs on Real-time Object Detection" (2023)
- Focal Loss: Lin et al., ICCV 2017
- Varifocal Loss: Zhang et al., CVPR 2021
- QGFL: Quality-Guided Focal Loss (this work, 2025)

---

**End of Integration Plan**
