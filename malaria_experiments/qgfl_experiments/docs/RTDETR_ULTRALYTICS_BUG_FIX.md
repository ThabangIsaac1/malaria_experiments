# RT-DETR Ultralytics Bug Fix - Critical Model Loading Issue

**Date:** 2025-10-07
**Status:** ✅ FIXED
**Severity:** CRITICAL - Causes 0% metrics for all RT-DETR evaluations

---

## Problem Summary

When RT-DETR models are loaded using the generic `YOLO()` class in Ultralytics, they produce **completely broken bounding box coordinates**, resulting in:
- ❌ 0% mAP, 0% precision, 0% recall on all evaluations
- ❌ Negative bounding box widths (e.g., width = -1.87)
- ❌ Corrupted coordinate values (e.g., x1=2.17, x2=0.30 for 2752x2192 image)
- ❌ Invalid class IDs (float 0.05 instead of integer 0 or 1)

---

## Root Cause Analysis

### Technical Details

**Issue:** Ultralytics `YOLO()` class assigns wrong predictor to RT-DETR models

```python
# BROKEN:
model = YOLO("rtdetr-l.pt")
# → Assigns: DetectionPredictor (for YOLO models)
# → Expected: RTDETRPredictor (for RT-DETR models)
```

**Consequence:** Wrong predictor uses incompatible postprocessing logic

| Aspect | RT-DETR with YOLO() (BROKEN) | RT-DETR with RTDETR() (FIXED) |
|--------|------------------------------|-------------------------------|
| Predictor | `DetectionPredictor` ❌ | `RTDETRPredictor` ✅ |
| Coordinates | `[2.17, 0.0, 0.29, 0.0]` ❌ | `[2129, 1744, 2315, 1925]` ✅ |
| Width | `-1.87` (negative!) ❌ | `186` (valid) ✅ |
| Class ID | `0.05` (float) ❌ | `0` (integer) ✅ |
| mAP@0.5 | `0%` ❌ | `68.68%` ✅ |

### Evidence - Coordinate Comparison

**Test image:** `003435d8-1da7-4501-b85d-8815b660ede4.jpg` (2752×2192 pixels)

**YOLO model (working correctly):**
```python
model = YOLO("yolov11s.pt")
results = model.predict(img)
box.xyxy[0] → tensor([2565.92, 2013.93, 2733.75, 2177.49])  # ✅ Valid pixel coords
box.conf → 0.915  # ✅ Valid confidence
box.cls → 0.0     # ✅ Integer class ID
```

**RT-DETR with YOLO() - BROKEN:**
```python
model = YOLO("rtdetr-l.pt")  # ❌ WRONG CLASS
results = model.predict(img)
box.xyxy[0] → tensor([2.1713, 0.0000, 0.2917, 0.0000])  # ❌ Corrupted coords
width = 0.2917 - 2.1713 = -1.8796  # ❌ NEGATIVE WIDTH!
box.conf → 0.8125                  # Looks OK
box.cls → 0.0502                   # ❌ Float instead of integer!
```

**RT-DETR with RTDETR() - FIXED:**
```python
model = RTDETR("rtdetr-l.pt")  # ✅ CORRECT CLASS
results = model.predict(img)
box.xyxy[0] → tensor([2129.09, 1743.82, 2314.87, 1925.13])  # ✅ Valid pixel coords
width = 2314.87 - 2129.09 = 185.78  # ✅ Positive width!
box.conf → 0.7791                    # ✅ Valid confidence
box.cls → 0                          # ✅ Integer class ID
```

---

## Related Ultralytics Issues

- **Issue #21684:** "RTDETR ONNX models incorrectly use YOLO validator causing dimension mismatch error"
  - Same root cause - wrong predictor/validator assignment
  - Affects ONNX exports and validation

---

## Solution

### Fix: Use `RTDETR()` class for RT-DETR models

```python
from ultralytics import YOLO, RTDETR

# Auto-detect and load correct class
def load_model(model_path, config):
    model_path_str = str(model_path)

    # Detect RT-DETR from path or config
    if ('rtdetr' in model_path_str.lower() or
        'rt-detr' in model_path_str.lower() or
        config.model_name.lower().startswith('rtdetr')):
        return RTDETR(model_path)  # ✅ Use RT-DETR class
    else:
        return YOLO(model_path)    # ✅ Use YOLO class
```

---

## Files Modified

### 1. `src/evaluation/evaluator.py`

**Before:**
```python
from ultralytics import YOLO

class ComprehensiveEvaluator:
    def __init__(self, model_path, ...):
        self.model = YOLO(model_path)  # ❌ Breaks RT-DETR
```

**After:**
```python
from ultralytics import YOLO, RTDETR

class ComprehensiveEvaluator:
    def __init__(self, model_path, dataset_path, config, ...):
        # Fix for RT-DETR: Use RTDETR class to get correct predictor
        if isinstance(model_path, (str, Path)):
            model_path_str = str(model_path)
            if ('rtdetr' in model_path_str.lower() or
                'rt-detr' in model_path_str.lower() or
                config.model_name.lower().startswith('rtdetr')):
                self.model = RTDETR(model_path)  # ✅ Correct for RT-DETR
            else:
                self.model = YOLO(model_path)    # ✅ Correct for YOLO
        else:
            self.model = model_path
```

**Impact:**
- ✅ Fixes `compute_global_metrics()` - model.val() now works
- ✅ Fixes `compute_per_class_metrics()` - model.predict() returns valid boxes
- ✅ Fixes `compute_pr_curves()` - valid coordinates for PR calculation
- ✅ Fixes `compute_stratified_analysis()` - prevalence analysis works
- ✅ Fixes `compute_error_analysis()` - TIDE analysis works
- ✅ Fixes `compute_confusion_matrix()` - confusion matrix works
- ✅ All visualizations now work correctly

### 2. `cluster_run_baseline.py`

**Multiple fixes applied:**

**A. RT-DETR Predictor Fix (Line ~870):**
```python
# Before:
model_eval = YOLO(best_model_path)

# After:
if config.model_name == 'rtdetr':
    model_eval = RTDETR(best_model_path)  # ✅ Correct predictor
else:
    model_eval = YOLO(best_model_path)    # ✅ Maintains YOLO functionality
```

**B. W&B Logging Fix (Line 306):**
```python
# Before:
'optimizer': config.optimizer,  # Was logging 'SGD' (config default)

# After:
'optimizer': args.optimizer,  # Logs 'auto' (actual parameter used)
```

**C. Config-Based Thresholds (Lines 887, 1007, 1063, 1703):**
```python
# Before:
results = model.predict(img, conf=0.5, iou=0.5, verbose=False)

# After:
results = model.predict(img, conf=config.conf, iou=config.iou, agnostic_nms=True, verbose=False)
```

**D. CLI Hyperparameter Support + Optimizer Auto-Selection (Lines 40-62):**
```python
# Added arguments:
parser.add_argument('--optimizer', default=None)  # None = auto-select based on model
parser.add_argument('--lr0', default=0.01)
parser.add_argument('--lrf', default=0.01)
parser.add_argument('--warmup-epochs', default=3.0)
parser.add_argument('--cls', default=0.5)
parser.add_argument('--box', default=7.5)

# CRITICAL FIX: Auto-select optimizer to preserve YOLO hyperparameters
if args.optimizer is None:
    if 'rtdetr' in args.model.lower():
        args.optimizer = 'auto'  # AdamW for RT-DETR (Ultralytics auto-selects)
    else:
        args.optimizer = 'SGD'   # Explicit SGD for YOLO (preserves our lr0/momentum)
```

**Why This Matters:**
- When `optimizer='auto'`, Ultralytics **ignores** user-specified `lr0` and `momentum`
- For YOLO, we want our custom hyperparameters (`lr0=0.01`, `momentum=0.95`)
- For RT-DETR, we want Ultralytics to auto-select AdamW with optimal defaults
- **Solution:** Use `optimizer='SGD'` for YOLO, `optimizer='auto'` for RT-DETR

**E. Auto GPU Detection (Line 544):**
```python
# Before:
'device': 'cpu',  # Force CPU

# After:
'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detect
```

### 3. `configs/baseline_config.py`

**Evaluation Threshold Update (Lines 41-43):**
```python
# Before:
conf: float = 0.5  # Generic threshold
iou: float = 0.5   # Generic threshold

# After:
conf: float = 0.25  # Guemas et al. methodology
iou: float = 0.45   # Domain-specific for malaria detection
```

**Rationale:** Guemas et al. (2024) paper on malaria detection uses conf≥0.25, IoU≥0.45, agnostic_nms=True for fair comparison across architectures.

### 4. `cluster_run_qgfl.py`

**Status:** ⏳ TODO - Same updates needed as cluster_run_baseline.py

---

## Validation Results

### Test Set Validation (5 sample images)

| Image | GT Boxes | Predictions | Coords Valid | Confidence |
|-------|----------|-------------|--------------|------------|
| 003435d8... | 77 | 197 | ✅ w=186, h=181 | 0.779 |
| 0118168a... | 92 | 229 | ✅ w=200, h=175 | 0.785 |
| 0202fe19... | 92 | 250 | ✅ w=176, h=179 | 0.817 |
| 071386d1... | 69 | 199 | ✅ w=167, h=169 | 0.754 |
| 083c00db... | 88 | 228 | ✅ w=170, h=179 | 0.754 |

**Result:** ✅ All coordinates valid on test set

### Validation Set Metrics (rtdetr_200.pt on D1)

| Metric | Before Fix | After Fix | Change |
|--------|------------|-----------|--------|
| mAP@0.5 | 0% ❌ | 68.68% ✅ | +68.68% |
| mAP@[0.5:0.95] | 0% ❌ | 54.64% ✅ | +54.64% |
| Precision | 0% ❌ | 52.93% ✅ | +52.93% |
| Recall | 0% ❌ | 70.63% ✅ | +70.63% |

**Result:** ✅ Evaluation completely fixed

---

## Impact on Different Evaluation Components

| Component | Method | Uses | Impact | Status |
|-----------|--------|------|--------|--------|
| Global Metrics | `model.val()` | Evaluator line 93 | ✅ FIXED | Returns valid metrics |
| Per-Class Metrics | `model.predict()` | Evaluator line 157+ | ✅ FIXED | Valid IoU matching |
| PR Curves | `model.predict()` | Evaluator line 261+ | ✅ FIXED | Valid precision/recall |
| Prevalence Analysis | `model.predict()` | Evaluator line 415+ | ✅ FIXED | Valid stratified recall |
| TIDE Error Analysis | `model.predict()` | Evaluator line 516+ | ✅ FIXED | Valid error classification |
| Confusion Matrix | `model.predict()` | Evaluator line 665+ | ✅ FIXED | Valid class matching |
| Visualizations | `model.predict()` | Evaluator line 728+ | ✅ FIXED | Valid bounding boxes |
| Decision Analysis | `model.predict()` | cluster_run line 2506+ | ⏳ TODO | Need to update |
| Inference Speed | `model.predict()` | cluster_run line 871 | ⏳ TODO | Need to update |

---

## Testing Protocol

### Before Deploying Any RT-DETR Experiments:

1. ✅ **Verify predictor class:**
   ```python
   model = RTDETR("model.pt")
   assert model.predictor.__class__.__name__ == "RTDETRPredictor"
   ```

2. ✅ **Verify coordinates:**
   ```python
   results = model.predict(img, conf=0.25)
   box = results.boxes[0]
   x1, y1, x2, y2 = box.xyxy[0].tolist()
   assert x2 > x1 and y2 > y1  # Positive width/height
   assert isinstance(int(box.cls.item()), int)  # Integer class ID
   ```

3. ✅ **Verify metrics:**
   ```python
   metrics = model.val(data=yaml, split='val', conf=0.25, iou=0.45)
   assert metrics.box.map50 > 0  # Non-zero mAP
   ```

---

## Why This Matters for Research

### Clinical Impact
- **Before:** RT-DETR appears completely unusable (0% recall)
- **After:** RT-DETR achieves 70.6% recall - clinically viable

### Fair Comparison
- **Before:** Can't compare YOLO vs RT-DETR (RT-DETR broken)
- **After:** Fair architecture comparison possible

### QGFL Validation
- **Before:** Can't validate QGFL works across architectures
- **After:** Can demonstrate QGFL improves both CNN and Transformer models

### Publication Quality
- **Before:** Reviewers would question 0% RT-DETR metrics
- **After:** Valid metrics support multi-architecture claims

---

## Smoke Test Validation

### 5-Epoch Smoke Test (W&B run: md7j0glq)
**Date:** 2025-10-08
**Config:** D1 binary, RT-DETR-L, conf=0.25, iou=0.45

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 1 | 66.8% | 29.9% | 17.2% | 12.3% |
| 2 | 35.0% | 38.0% | 36.0% | 28.9% |
| 3 | 88.8% | 43.4% | 42.3% | 34.7% |
| 4 | 59.1% | 79.1% | 64.7% | 51.9% |
| **5** | **59.7%** | **81.8%** | **66.4%** | **52.9%** |

**Validation:**
- ✅ All coordinates valid (positive widths/heights)
- ✅ Integer class IDs (no float 0.05 errors)
- ✅ Non-zero metrics (fix confirmed working)
- ✅ W&B logging accurate (`optimizer: "auto"` now correct)
- ✅ Config-based thresholds working (conf=0.25, iou=0.45)

### 15-Epoch Smoke Test (W&B run: hgd51uid)
**Date:** 2025-10-07-08
**Config:** D1 binary, RT-DETR-L, conf=0.25, iou=0.45

| Epoch | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| 5 | 56.5% | 75.2% | 63.9% | 51.2% |
| 10 | 71.2% | 78.1% | 71.5% | 58.9% |
| **15** | **74.0%** | **76.7%** | **73.9%** | **61.3%** |

**Result:** ✅ Training stable, metrics improving, ready for cluster deployment

---

## YOLO Compatibility Verification

**Critical Question:** Do RT-DETR fixes break YOLO models?

### Code Analysis:

**1. evaluator.py (Lines 20-28):**
```python
if 'rtdetr' in model_path_str.lower() or 'rt-detr' in model_path_str.lower():
    self.model = RTDETR(model_path)  # Only for RT-DETR
else:
    self.model = YOLO(model_path)    # ✅ YOLO unchanged
```
**Result:** ✅ YOLO models still use `YOLO()` class - no predictor change

**2. cluster_run_baseline.py (Line 873):**
```python
if config.model_name == 'rtdetr':
    model_eval = RTDETR(best_model_path)  # Only for RT-DETR
else:
    model_eval = YOLO(best_model_path)    # ✅ YOLO unchanged
```
**Result:** ✅ YOLO models still use `YOLO()` class - no predictor change

**3. Evaluation Thresholds:**
```python
# Before: conf=0.5, iou=0.5 (both YOLO and RT-DETR)
# After:  conf=0.25, iou=0.45 (both YOLO and RT-DETR)
```
**Impact:** ⚠️ Lower thresholds mean more detections for BOTH architectures
- Increases recall (captures more parasites)
- May decrease precision slightly
- Matches Guemas et al. (2024) methodology for fair comparison
- **Requires re-running YOLO baselines with new thresholds for fair comparison**

**4. Optimizer Selection:**
```python
# Ultralytics auto-selection logic (when optimizer='auto'):
# iterations = epochs × batches_per_epoch
# if iterations > 10,000: optimizer = SGD
# else: optimizer = AdamW

# YOLO v8/v11 @ 200 epochs:
# 200 epochs × 150 batches = 30,000 iterations → SGD ✅

# YOLO v8/v11 @ 3 epochs (smoke test):
# 3 epochs × 15 batches = 45 iterations → AdamW ⚠️

# RT-DETR @ 200 epochs:
# 200 epochs × 15 batches = 3,000 iterations → AdamW ✅

# RT-DETR @ 5 epochs (smoke test):
# 5 epochs × 15 batches = 75 iterations → AdamW ✅
```
**Result:** ✅ Optimizer selection unchanged for both architectures
**Important:** Smoke tests use AdamW for both models (low iteration count). Cluster 200-epoch runs will use correct optimizers (SGD for YOLO, AdamW for RT-DETR)

**5. W&B Logging + Optimizer Display:**
```python
# Before: 'optimizer': config.optimizer  # Logged 'SGD' (wrong for RT-DETR)
# After:  'optimizer': args.optimizer    # Logs actual parameter passed
```
**Result:** W&B now logs the **actual parameter passed**, not config default

**What You'll See in W&B:**
- **YOLO runs:** `optimizer: "SGD"` ✅ (our explicit choice to preserve hyperparameters)
- **RT-DETR runs:** `optimizer: "auto"` ✅ (Ultralytics selects AdamW)

**Understanding the Values:**

| W&B Value | Meaning | YOLO Behavior | RT-DETR Behavior |
|-----------|---------|---------------|------------------|
| `"SGD"` | Explicit SGD | Uses our lr0=0.01, momentum=0.95 | N/A |
| `"auto"` | Ultralytics auto-select | Would select based on iterations | Selects AdamW with optimal LR |

**CRITICAL: Why YOLO Uses Explicit SGD**

When `optimizer='auto'`, Ultralytics **ignores** user-specified `lr0` and `momentum`:
```
optimizer:'auto' found, ignoring 'lr0=0.01' and 'momentum=0.95'...
```

- **For YOLO:** We want our custom `lr0=0.01`, `momentum=0.95` → Use `optimizer='SGD'`
- **For RT-DETR:** We want Ultralytics to calculate optimal AdamW params → Use `optimizer='auto'`

**How to verify actual optimizer in console:**
- YOLO: `optimizer: SGD(lr=0.01, momentum=0.95)` ✅
- RT-DETR: `optimizer: AdamW(lr=0.001667, momentum=0.9)` ✅

### Conclusion:
**✅ All RT-DETR fixes are architecture-specific or beneficial to both models**
- Predictor selection: RT-DETR-specific logic with YOLO fallback
- Evaluation thresholds: Applied equally to both (Guemas methodology)
- W&B logging: More accurate for both architectures
- Optimizer: Ultralytics auto-selection unchanged

**⚠️ Action Required:** Run YOLO smoke test to verify no breakage

---

## Next Steps

1. ✅ Update `evaluator.py` (DONE)
2. ✅ Update `cluster_run_baseline.py` (DONE)
3. ✅ Update `configs/baseline_config.py` (DONE)
4. ✅ Run RT-DETR smoke tests (5 & 15 epochs) (DONE - PASSED)
5. ⏳ **CRITICAL:** Run 3-epoch YOLO smoke test to verify no breakage
6. ⏳ Update `cluster_run_qgfl.py` with same fixes
7. ⏳ Upload updated files to cluster
8. ⏳ Deploy baselines to cluster (D1-D3, YOLO v8/v11, RT-DETR, 200 epochs)
9. ⏳ Deploy QGFL runs to cluster after baseline validation

---

## Key Takeaway

**Always use `RTDETR()` class for RT-DETR models, never `YOLO()` class.**

This is an Ultralytics framework limitation, not our codebase issue. The fix ensures correct predictor assignment and valid bounding box postprocessing.
