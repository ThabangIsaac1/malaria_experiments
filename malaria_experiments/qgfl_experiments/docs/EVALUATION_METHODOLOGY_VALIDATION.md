# Evaluation Methodology Validation - Sound & Standard Practices

**Date:** 2025-10-07
**Purpose:** Document that our evaluation methodology follows sound ML/CV practices

---

## Evaluation Components & Predictor Usage

### **Where We Use model.predict()** (Per-Image Evaluation)

All per-image analyses use `model.predict()` with **consistent parameters**:
```python
results = model.predict(
    img_path, 
    conf=0.25,      # Guemas et al. methodology
    iou=0.45,       # Guemas et al. methodology  
    agnostic_nms=True,  # Class-agnostic NMS (Guemas)
    verbose=False
)
```

**Applied in:**
1. ✅ **Per-class metrics** (`evaluator.py` line 157)
2. ✅ **Precision-Recall curves** (`evaluator.py` line 261)
3. ✅ **Prevalence-stratified analysis** (`evaluator.py` line 415)
4. ✅ **TIDE error analysis** (`evaluator.py` line 516)
5. ✅ **Confusion matrix** (`evaluator.py` line 665)
6. ✅ **Visualizations** (`evaluator.py` line 728)
7. ✅ **Decision analysis** (`cluster_run_baseline.py` line 2506-2507)
8. ✅ **Inference speed** (`cluster_run_baseline.py` line 871)

### **Where We Use model.val()** (Dataset-Level Evaluation)

Global metrics use `model.val()` - Ultralytics' official validation method:
```python
metrics = model.val(
    data=yaml_path,
    split='test',
    conf=0.25,
    iou=0.45,
    agnostic_nms=True,
    verbose=False
)
```

**Applied in:**
1. ✅ **Global metrics** (`evaluator.py` line 93)
   - mAP@0.5, mAP@[0.5:0.95]
   - Overall precision/recall
   - Per-class mAP@[0.5:0.95]

---

## Why This Is Sound Methodology

### 1. **Separation of Concerns**
- `model.val()` → Official Ultralytics validation (handles batching, metrics aggregation)
- `model.predict()` → Fine-grained control for custom analyses

### 2. **Consistency Across Methods**
- **Same thresholds everywhere:** conf=0.25, iou=0.45, agnostic_nms=True
- **Same matching logic:** IoU-based matching in all analyses
- **Same NMS:** Class-agnostic to prevent overlapping different classes

### 3. **Alignment with Literature**

**Guemas et al. 2024 methodology:**
- Confidence threshold: ≥0.25 ✅ (we use 0.25)
- IoU threshold: ≥0.45 ✅ (we use 0.45)
- Class-agnostic NMS: Yes ✅ (we use True)
- Applied to: ALL models (YOLO, RT-DETR) ✅ (we apply to all)

**Standard object detection evaluation:**
- COCO metrics: mAP@[0.5:0.95] ✅ (computed by model.val())
- Pascal VOC metrics: mAP@0.5 ✅ (computed by model.val())
- Per-class analysis: Standard ✅ (manual IoU matching)
- PR curves: Standard ✅ (threshold sweep from 0.01-1.0)

### 4. **Domain-Specific Appropriateness**

**Malaria Detection Requirements:**
- High sensitivity (catch infected cells) → Lower conf threshold (0.25) ✅
- Precise localization (small cells) → Domain-tuned IoU (0.45) ✅
- Mutually exclusive classes → Agnostic NMS (True) ✅
- Clinical interpretability → Decision analysis with uncertainty ✅

---

## Parameter Justification

### Training Parameters (Architecture-Specific)

**YOLO:**
- Optimizer: SGD (standard for CNNs)
- LR: 0.01 → 0.01 (constant)
- Momentum: 0.937 (YOLOv8 default)
- NMS IoU (training): 0.7 (prevents duplicate anchors)

**RT-DETR:**
- Optimizer: AdamW (required for transformers)
- LR: 0.01 → 0.01 * 0.01 (learning rate decay)
- Warmup: 3 epochs (transformer warmup)
- NMS IoU (training): 0.7 (same as YOLO)

**Rationale:** Different architectures need different optimizers/schedules

### Evaluation Parameters (Methodology-Specific)

**ALL MODELS (YOLO + RT-DETR):**
- Conf: 0.25 (Guemas methodology)
- IoU: 0.45 (Guemas methodology)
- Agnostic NMS: True (Guemas methodology)

**Rationale:** Fair comparison requires same evaluation criteria

### QGFL Parameters (Our Innovation)

**Loss-specific (not evaluation):**
- gamma_infected: 8.0 (high focus on hard infected cells)
- gamma_uninfected: 4.0 (moderate focus on uninfected)
- uiou_start: 2.0 (prioritize localization early)
- uiou_end: 0.5 (balance loc/class late)

**Rationale:** From our QGFL paper - class imbalance mitigation

---

## Validation of Predictor Outputs

### Test: YOLO Model
```python
model = YOLO("yolov11s.pt")
results = model.predict(img, conf=0.25, iou=0.45, agnostic_nms=True)
```
- ✅ Predictor: `DetectionPredictor` (correct)
- ✅ Coordinates: Absolute pixels [2565, 2014, 2734, 2177]
- ✅ Confidence: 0.915 (valid range 0-1)
- ✅ Class ID: 0 (integer 0 or 1)

### Test: RT-DETR Model
```python
model = RTDETR("rtdetr-l.pt")  # NOT YOLO()!
results = model.predict(img, conf=0.25, iou=0.45, agnostic_nms=True)
```
- ✅ Predictor: `RTDETRPredictor` (correct)
- ✅ Coordinates: Absolute pixels [2129, 1744, 2315, 1925]
- ✅ Confidence: 0.779 (valid range 0-1)
- ✅ Class ID: 0 (integer 0 or 1)

**Critical:** Must use `RTDETR()` class, not `YOLO()` - see RTDETR_ULTRALYTICS_BUG_FIX.md

---

## Reproducibility Checklist

### Must Log to W&B:

**Training hyperparameters:**
- ✅ Optimizer (SGD/AdamW)
- ✅ Learning rate (lr0, lrf)
- ✅ Warmup epochs
- ✅ Loss weights (box, cls, dfl)
- ✅ Batch size

**Evaluation thresholds:**
- ⏳ **TODO:** conf = 0.25
- ⏳ **TODO:** iou = 0.45
- ⏳ **TODO:** agnostic_nms = True

**QGFL parameters (QGFL experiments only):**
- ⏳ **TODO:** gamma_infected = 8.0
- ⏳ **TODO:** gamma_uninfected = 4.0
- ⏳ **TODO:** uiou_start = 2.0
- ⏳ **TODO:** uiou_end = 0.5

---

## Comparison with Other Works

| Paper | Conf Threshold | IoU Threshold | NMS | Fair? |
|-------|----------------|---------------|-----|-------|
| **Guemas et al. 2024** | 0.25 | 0.45 | Agnostic | ✅ All models same |
| **Our QGFL paper** | Not specified | Not specified | Default | ❌ Gap to fill |
| **Davidson et al.** | Not specified | 0.5 (implied) | Default | ⚠️ Unclear |
| **This work** | 0.25 | 0.45 | Agnostic | ✅ All models same |

**Our contribution:** Explicit methodology, clinically validated thresholds, fair multi-architecture comparison

---

## Summary

**✅ Our evaluation methodology is sound because:**
1. Uses standard Ultralytics API (model.val + model.predict)
2. Consistent thresholds across all components
3. Follows published methodology (Guemas et al. 2024)
4. Separates training (architecture-specific) from evaluation (methodology-specific)
5. Validated outputs match expected format
6. Clinically appropriate for malaria detection

**⏳ Action needed:**
- Add evaluation thresholds to W&B logging
- Add QGFL parameters to W&B logging (QGFL experiments only)

