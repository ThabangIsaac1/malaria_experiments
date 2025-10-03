# CRITICAL AUDIT FINDINGS - train_baseline.py

## Issue: Script Missing 90% of Notebook Evaluation Components!

### Notebook vs Script Comparison

| Component | Notebook (3,171 lines) | Script (575 lines) | Status |
|-----------|----------------------|-------------------|--------|
| **Training** | ✅ Cells 1-8 | ✅ Lines 1-575 | **COMPLETE** |
| **Basic Evaluation** | ✅ Cell 10-12 | ✅ Lines 497-528 | **PARTIAL** |
| **Per-Class Analysis** | ✅ Cell 13A-13B | ❌ MISSING | **MISSING** |
| **Prevalence-Stratified** | ✅ Cell 14 (200 lines) | ❌ MISSING | **MISSING** |
| **PR Curves** | ✅ Cell 15 (250 lines) | ❌ MISSING | **MISSING** |
| **TIDE Error Analysis** | ✅ Cell 16 (350 lines) | ❌ MISSING | **MISSING** |
| **GT vs Predictions Viz** | ✅ Cell 17 (300 lines) | ❌ MISSING | **MISSING** |
| **Decision Analysis** | ✅ Cell 18 (500 lines) | ❌ MISSING | **MISSING** |
| **Comprehensive W&B Logging** | ✅ Cell 19 (600 lines) | ❌ MISSING | **MISSING** |

---

## What's Currently in train_baseline.py

### ✅ **INCLUDED (Working)**
1. **Dataset Setup**
   - Auto COCO→YOLO conversion
   - YAML generation
   - Class distribution calculation

2. **Training Pipeline**
   - Model loading (YOLOv8s/YOLOv11s)
   - Strategy selection (NoWeights/QGFL)
   - YOLO training with full hyperparameters
   - Training curves logged to W&B

3. **Basic Evaluation**
   ```python
   evaluator = ComprehensiveEvaluator(...)
   eval_results = evaluator.evaluate_model('test')
   # Logs: precision, recall, mAP50, mAP50-95
   ```

---

## ❌ **MISSING CRITICAL COMPONENTS**

### 1. **Per-Class Performance Analysis** (Notebook Cell 13)
**What it does:**
- Breaks down metrics per class (Uninfected vs Infected)
- Calculates F1, precision, recall for each class
- Identifies minority class performance gaps

**Why critical:**
- Essential for understanding class imbalance impact
- Core metric for QGFL paper contribution
- Shows if model ignores minority class

**Missing from script:** YES

---

### 2. **Prevalence-Stratified Analysis** (Notebook Cell 14 - 200 lines)
**What it does:**
```python
# Bins images by infection density:
'0-1%':   Very low parasitemia (hardest to detect)
'1-3%':   Low parasitemia (clinically critical range)
'3-5%':   Moderate
'>5%':    High (easier to detect)

# Calculates recall per bin
# THIS IS THE KEY METRIC FROM THE QGFL PAPER!
```

**From QGFL paper:**
> "QGFL achieves remarkable improvement in detecting infected cells in the **clinically vital 1–3% parasitaemia range**"

**Why critical:**
- **THIS IS THE PRIMARY CONTRIBUTION OF THE PAPER**
- Without this, you can't claim QGFL improvements
- Baseline needs this metric to compare against

**Missing from script:** YES - CRITICAL!

---

### 3. **TIDE Error Analysis** (Notebook Cell 16 - 350 lines)
**What it does:**
- Classifies detection errors into 6 categories:
  - **Classification errors:** GT detected but wrong class
  - **Localization errors:** Correct class but bad bbox (IoU < 0.5)
  - **Duplicate detections:** Multiple predictions for same GT
  - **Background errors:** False positives (no GT)
  - **Missed detections:** GT not detected at all
  - **Other errors**

**Why critical:**
- Identifies WHERE the model fails
- Guides QGFL improvements (e.g., "focus on missed detections")
- Standard evaluation for object detection papers

**Missing from script:** YES

---

### 4. **Precision-Recall Curves** (Notebook Cell 15 - 250 lines)
**What it does:**
- Plots full PR curves for each class
- Calculates AP (area under PR curve)
- Shows confidence threshold trade-offs

**Why critical:**
- Visual representation of model performance
- Required for publication figures
- Shows optimal threshold selection

**Missing from script:** YES

---

### 5. **Ground Truth vs Predictions Visualization** (Notebook Cell 17 - 300 lines)
**What it does:**
- Side-by-side comparison of GT and predictions
- Color-coded by confidence level
- Saves visual examples to disk

**Why critical:**
- Qualitative analysis for paper figures
- Shows actual model behavior on images
- Identifies failure modes visually

**Missing from script:** YES

---

### 6. **Decision Analysis** (Notebook Cell 18 - 500 lines)
**What it does:**
- Analyzes True Positives, False Positives, False Negatives
- Breaks down by confidence levels (Low/Medium/High)
- Calculates precision/recall at different thresholds

**Why critical:**
- Deep dive into model decision-making
- Shows calibration issues
- Identifies confidence threshold problems

**Missing from script:** YES

---

### 7. **Comprehensive W&B Logging** (Notebook Cell 19 - 600 lines)
**What it does:**
- Logs ALL evaluation metrics to W&B:
  - Per-class tables
  - Prevalence-stratified tables
  - TIDE error charts
  - PR curve plots
  - Confusion matrices
  - Sample visualizations

**Why critical:**
- Makes results accessible in W&B dashboard
- Enables comparison across experiments
- Required for collaborative analysis

**Missing from script:** YES

---

## Impact Assessment

### **Current State**
```
train_baseline.py:
├── Training: ✅ COMPLETE (100%)
└── Evaluation: ⚠️ MINIMAL (10%)
    ├── Global metrics only (mAP, precision, recall)
    └── No per-class, no stratified, no TIDE, no visualizations
```

### **Problem**
1. **Can't validate baseline properly** - Missing key metrics
2. **Can't compare to QGFL later** - No prevalence-stratified baseline
3. **Can't write paper** - Missing required analyses
4. **Can't debug failures** - No TIDE or decision analysis

---

## Recommended Solutions

### **Option A: Add Full Evaluation to train_baseline.py** (NOT Recommended)
- Pros: Single script does everything
- Cons:
  - Script becomes 2,500+ lines
  - Slower (adds 30-60min per experiment)
  - Harder to maintain
  - Evaluation can't be re-run independently

### **Option B: Separate evaluate_baseline.py Script** (RECOMMENDED)
- Pros:
  - Clean separation: train.py (fast) + evaluate.py (comprehensive)
  - Can re-run evaluation without retraining
  - Easier to maintain
  - Cluster: run training jobs first, then evaluation locally
- Cons:
  - Two scripts to manage
  - Need to pass model path between them

### **Option C: Make Evaluation Optional in train_baseline.py**
- Pros:
  - Single script
  - `--quick-eval` for basic metrics
  - `--full-eval` for everything
- Cons:
  - Still large script
  - Harder to debug

---

## Immediate Action Required

**Before Cluster Submission:**

1. **Create `evaluate_baseline.py`** with ALL notebook evaluation:
   - Per-class analysis
   - **Prevalence-stratified analysis** (CRITICAL)
   - TIDE error analysis
   - PR curves
   - GT vs Predictions visualizations
   - Decision analysis
   - Comprehensive W&B logging

2. **Update `train_baseline.py`** to:
   - Add `--skip-eval` flag (default: True for cluster speed)
   - Keep basic evaluation for quick validation
   - Save model path to JSON for later evaluation

3. **Create `run_full_pipeline.sh`**:
   ```bash
   # Cluster: Train only (fast)
   python train_baseline.py --model yolov8s --dataset d1 --skip-eval

   # Local: Full evaluation (comprehensive)
   python evaluate_baseline.py --model-path runs/detect/.../weights/best.pt
   ```

4. **Create smoke test**:
   ```bash
   # Test training + evaluation on tiny subset
   python smoke_test.py --model yolov8s --dataset d1 --epochs 1
   ```

---

## What to Do Now

**DECISION POINT:**

**Question 1:** Do you want:
- A) Separate `evaluate_baseline.py` script (recommended)
- B) Add full evaluation to `train_baseline.py`
- C) Hybrid approach (basic in training, comprehensive separate)

**Question 2:** For cluster submission:
- Run training only (fast, ~24hrs/job), then evaluate locally?
- Run training + full evaluation (slower, ~48hrs/job)?

**Question 3:** Priority:
- Fix this before any cluster submission?
- Or run limited training first, add evaluation later?

---

## Bottom Line

**The current `train_baseline.py` is only 20% of what the notebook does.**

It will train models fine, but you'll be **missing 80% of the analysis** you need for:
- Understanding baseline performance
- Comparing to QGFL later
- Writing the paper
- Debugging issues

**We need to decide how to handle the full evaluation pipeline before cluster deployment.**
