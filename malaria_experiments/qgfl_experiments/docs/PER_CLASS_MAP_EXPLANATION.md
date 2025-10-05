# Per-Class mAP Explanation
**Date**: October 4, 2025
**Purpose**: Clarify per-class mAP metrics for research reporting

---

## Quick Answer

**Do you have per-class mAP?**
- ✅ YES - Just added per-class mAP@50-95
- ❌ NO - Per-class mAP@50 (not available from YOLO)

**What you now have:**
```
Global Metrics:
├─ Overall mAP@50        ✅
├─ Overall mAP@50-95     ✅
└─ Per-class mAP@50-95   ✅ (just added)

Per-Class Custom Metrics:
├─ Precision at conf=0.5 ✅
├─ Recall at conf=0.5    ✅
├─ F1 at conf=0.5        ✅
└─ TP/FP/FN              ✅
```

---

## Detailed Explanation

### **What is mAP vs Per-Class Metrics?**

#### **1. mAP (Mean Average Precision)**

**What it measures:**
- Model's ability to detect objects across ALL confidence thresholds
- Sweeps from conf=0.0 to conf=1.0
- Computes precision-recall curve
- Takes area under curve

**Variants:**
```
mAP@50:
├─ IoU threshold: 0.5
├─ A detection is "correct" if IoU > 0.5
└─ More lenient (easier to achieve)

mAP@50-95:
├─ IoU thresholds: 0.5, 0.55, 0.60, ... 0.95
├─ Average across all these IoU levels
└─ More strict (requires precise localization)
```

**Example:**
```
Model predicts infected cell:
├─ At conf=0.9: Precision=95%, Recall=50%
├─ At conf=0.5: Precision=80%, Recall=70%
├─ At conf=0.1: Precision=60%, Recall=90%
└─ mAP = Area under this curve = 75%
```

#### **2. Per-Class Metrics (Your Custom Implementation)**

**What it measures:**
- Model's performance at a SINGLE operating threshold
- Fixed conf=0.5 (your clinical threshold)
- Actual TP, FP, FN counts

**Example:**
```
At conf=0.5:
├─ TP: 89 infected cells detected correctly
├─ FP: 15 false alarms
├─ FN: 214 infected cells missed
├─ Precision: 89/(89+15) = 85.6%
├─ Recall: 89/(89+214) = 29.4%
└─ F1: 2*85.6*29.4/(85.6+29.4) = 43.6%
```

---

## Why Both Matter

### **mAP (Benchmark Comparison):**

**Used for:**
```
✅ Comparing to other papers
✅ Showing model capability across thresholds
✅ Standard object detection metric
✅ Required for COCO-style evaluations
```

**Example use in paper:**
```
"Our RT-DETR baseline achieved 75.2% mAP@50 and
55.8% mAP@50-95 on D2, outperforming YOLOv8s
(70.1% mAP@50) by 5.1 percentage points."
```

### **Per-Class P/R/F1 at conf=0.5 (Clinical Relevance):**

**Used for:**
```
✅ Clinical deployment threshold
✅ Single operating point
✅ Actual counts (TP/FP/FN)
✅ Cost-benefit analysis (FP vs FN trade-off)
```

**Example use in paper:**
```
"At the clinical operating threshold (conf=0.5),
our model achieved 85.6% precision and 29.4% recall
on infected cells, indicating high specificity but
low sensitivity for low-parasitemia cases."
```

---

## What YOLO Provides

### **Global (Overall) Metrics:**

```python
metrics = model.val(...)

metrics.box.map50      → Overall mAP@50 (single number)
metrics.box.map        → Overall mAP@50-95 (single number)
metrics.box.mp         → Overall precision
metrics.box.mr         → Overall recall
```

### **Per-Class Metrics:**

```python
metrics.box.maps       → Per-class mAP@50-95 (array)
                         Example: [0.852, 0.721]
                         [Uninfected mAP, Infected mAP]
```

### **What YOLO Does NOT Provide:**

```
❌ Per-class mAP@50 separately
   (Only provides overall mAP@50 + per-class mAP@50-95)

❌ Per-class precision/recall at specific threshold
   (You compute this yourself in compute_per_class_metrics)

❌ Per-class TP/FP/FN counts
   (You compute this yourself)
```

---

## Your Complete Metric Suite

### **After Today's Addition:**

```
Global Metrics (from YOLO):
├─ Overall mAP@50         ✅ Standard benchmark
├─ Overall mAP@50-95      ✅ Strict localization
├─ Overall Precision      ✅ Average across classes
├─ Overall Recall         ✅ Average across classes
└─ Per-class mAP@50-95    ✅ (JUST ADDED)

Per-Class Custom Metrics (your implementation):
├─ Precision at conf=0.5  ✅ Clinical threshold
├─ Recall at conf=0.5     ✅ Clinical threshold
├─ F1 at conf=0.5         ✅ Harmonic mean
├─ TP/FP/FN counts        ✅ Actual detections
└─ Support                ✅ Ground truth count

Additional Analyses:
├─ Confusion Matrix       ✅ Class confusion
├─ TIDE Error Analysis    ✅ Error categorization
├─ PR Curves              ✅ Full curve data
└─ Prevalence-Stratified  ✅ By parasitemia bins
```

---

## Example Results Table (After 200 Epochs)

### **Global Metrics:**

```
Dataset: D2 (P. vivax)
Model: YOLOv8s

╔════════════════════════╤═══════╗
║ Metric                 │ Value ║
╠════════════════════════╪═══════╣
║ Overall mAP@50         │ 70.1% ║
║ Overall mAP@50-95      │ 52.3% ║
║ Overall Precision      │ 79.2% ║
║ Overall Recall         │ 65.8% ║
╚════════════════════════╧═══════╝
```

### **Per-Class mAP@50-95 (NEW):**

```
╔════════════════════════╤═══════╗
║ Class                  │  mAP  ║
╠════════════════════════╪═══════╣
║ Uninfected             │ 85.2% ║ ← Easier class
║ Infected               │ 72.1% ║ ← Harder class
╚════════════════════════╧═══════╝

Note: These are averaged across IoU 0.5-0.95
```

### **Per-Class Metrics at conf=0.5 (EXISTING):**

```
╔════════════════╤═══════╤════════╤═══════╤═════════╗
║ Class          │   P   │   R    │  F1   │ Support ║
╠════════════════╪═══════╪════════╪═══════╪═════════╣
║ Uninfected     │ 88.3% │ 93.8%  │ 90.9% │  5,059  ║
║ Infected       │ 85.6% │ 29.4%  │ 43.6% │    303  ║
╚════════════════╧═══════╧════════╧═══════╧═════════╝

Note: These are at fixed conf=0.5 (your clinical threshold)
```

---

## Interpretation

### **Example: Infected Class**

```
Per-Class mAP@50-95 = 72.1%
├─ What it means: Across all confidence thresholds,
│   model achieves 72.1% average precision
├─ Good for: Benchmark comparison
└─ Doesn't tell: Performance at your operating threshold

Per-Class Recall at conf=0.5 = 29.4%
├─ What it means: At conf=0.5, model detects
│   only 29.4% of infected cells
├─ Good for: Clinical deployment decision
└─ Shows: Model struggles with this class at conf=0.5
```

**Why the difference?**
```
mAP@50-95 = 72.1% (good overall capability)
Recall@0.5 = 29.4% (poor at clinical threshold)

Explanation:
├─ At lower thresholds (conf=0.1): High recall ~70%
├─ At clinical threshold (conf=0.5): Low recall 29.4%
└─ Model CAN detect infected cells but lacks confidence

This tells you:
├─ Model has the capability (mAP is decent)
├─ But needs confidence calibration or
├─ More training on hard cases (rings, low parasitemia)
└─ QGFL should help with this!
```

---

## For Your Research Paper

### **Recommended Reporting:**

**1. Results Table:**
```
Table 1: Baseline Performance on D1, D2, D3

Model      | Dataset | mAP@50 | mAP@50-95 | Infected Recall@0.5
-----------|---------|--------|-----------|--------------------
YOLOv8s    | D1      | 78.2%  | 58.1%     | 72.5%
YOLOv8s    | D2      | 70.1%  | 52.3%     | 29.4% ← Problem!
YOLOv8s    | D3      | 82.5%  | 61.8%     | 68.3%
RT-DETR    | D1      | 79.1%  | 59.3%     | 74.2%
RT-DETR    | D2      | 75.2%  | 55.8%     | 42.1% ← Better!
RT-DETR    | D3      | 85.3%  | 64.2%     | 72.8%
```

**2. Per-Class Breakdown (Appendix):**
```
Table A1: Per-Class Performance (D2, YOLOv8s)

Class      | mAP@50-95 | P@0.5 | R@0.5 | F1@0.5 | Support
-----------|-----------|-------|-------|--------|--------
Uninfected | 85.2%     | 88.3% | 93.8% | 90.9%  | 5,059
Infected   | 72.1%     | 85.6% | 29.4% | 43.6%  | 303
```

**3. Discussion:**
```
"While our baseline YOLOv8s achieved competitive
mAP@50-95 (52.3%) on D2, indicating good overall
detection capability, the recall at our clinical
operating threshold (conf=0.5) was only 29.4% for
infected cells. This discrepancy suggests the model
can detect infected cells but lacks confidence in its
predictions, particularly for challenging cases such
as ring-stage parasites. We hypothesize that QGFL's
quality-guided loss weighting will improve confidence
calibration for these hard samples."
```

---

## Summary

**What you asked:**
> Do I have per-class mAP@50 and mAP@50-95?

**Answer:**
```
Per-class mAP@50:       ❌ Not available from YOLO
Per-class mAP@50-95:    ✅ Just added (metrics.box.maps)
Per-class P/R/F1@0.5:   ✅ You already had this
```

**Is this sufficient?**
```
✅ YES - Your metrics are comprehensive

You have:
1. Global mAP (overall performance)
2. Per-class mAP@50-95 (benchmark per class)
3. Per-class P/R/F1 at conf=0.5 (clinical relevance)
4. Confusion matrix (class relationships)
5. TIDE errors (error analysis)
6. Prevalence stratification (parasitemia bins)

This is MORE than most papers report!
```

**Advantage over other work:**
```
Most papers report ONLY:
├─ Overall mAP
└─ Maybe per-class mAP

You report:
├─ Overall mAP ✅
├─ Per-class mAP ✅
├─ Per-class P/R/F1 at clinical threshold ✅
├─ Confusion matrix ✅
├─ Error analysis ✅
├─ Prevalence stratification ✅
└─ Much more comprehensive! ⭐
```
