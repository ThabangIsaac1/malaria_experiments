# Solution: Single-Script Architecture with W&B Integration

## Decision: Keep Everything in `train_baseline.py` ✅

### Why This is Correct

**W&B Best Practice:**
```python
# CORRECT (Notebook approach)
wandb.init()           # Cell 7
model.train()          # Cell 8
evaluate_comprehensive()  # Cells 10-18
wandb.log({...})       # Cell 19 - ALL metrics
wandb.finish()         # End

# WRONG (Separated approach)
# Script 1:
wandb.init()
model.train()
wandb.finish()  # ← Closes run!

# Script 2:
wandb.init()    # ← New run! Different from training
evaluate()
wandb.finish()
```

**The Problem with Separation:**
- Creates **2 different W&B runs**
- Training metrics in Run A
- Evaluation metrics in Run B
- Can't compare them side-by-side
- Breaks the "one experiment = one run" principle

---

## Implementation Strategy

### Enhanced `train_baseline.py` Structure

```python
def train(args):
    # 1. Setup (already done)
    # 2. Initialize W&B (already done)
    # 3. Train model (already done)

    # 4. Comprehensive Evaluation (NEW - add from notebook)
    if not args.quick_mode:
        # Cell 10-12: Basic evaluation (already have basic version)
        # Cell 13: Per-class analysis (ADD)
        # Cell 14: Prevalence-stratified analysis (ADD - CRITICAL!)
        # Cell 15: PR curves (ADD)
        # Cell 16: TIDE errors (ADD)
        # Cell 17: Visualizations (ADD)
        # Cell 18: Decision analysis (ADD)

    # 5. Comprehensive W&B Logging (NEW - add from Cell 19)
    if args.use_wandb:
        log_comprehensive_metrics()  # All charts, tables, artifacts

    # 6. Finish W&B (already done)
    wandb.finish()
```

### Flags for Flexibility

```bash
# Full pipeline (for final runs)
python train_baseline.py --model yolov8s --dataset d1 --epochs 200

# Quick mode (for cluster speed or debugging)
python train_baseline.py --model yolov8s --dataset d1 --epochs 200 --quick-mode

# Just evaluation (re-run analysis on existing model)
python train_baseline.py --eval-only --model-path runs/.../best.pt
```

---

## What to Add to train_baseline.py

### Priority 1: CRITICAL (Must have before cluster)

**1. Prevalence-Stratified Analysis** (~100 lines)
```python
def prevalence_stratified_analysis(evaluator, split='test'):
    """Bin images by infection density and calculate recall per bin"""
    bins = {'0-1%': [], '1-3%': [], '3-5%': [], '>5%': []}
    # Calculate per-bin recall
    # THIS IS YOUR PAPER'S KEY METRIC!
    return stratified_results
```

**2. Per-Class Metrics** (~50 lines)
```python
def per_class_analysis(eval_results):
    """Break down precision/recall/F1 per class"""
    for class_name in class_names:
        # Calculate class-specific metrics
        # Log to W&B tables
    return per_class_results
```

**3. Comprehensive W&B Logging** (~200 lines from Cell 19)
```python
def log_comprehensive_wandb(train_results, eval_results, stratified, ...):
    """Log all metrics in organized W&B structure"""
    # Charts: wandb.log({'charts/test/mAP50': ...})
    # Tables: wandb.log({'tables/per_class': wandb.Table(...)})
    # Artifacts: wandb.log_artifact(...)
```

### Priority 2: Important (Good to have)

**4. TIDE Error Analysis** (~150 lines)
```python
def compute_tide_errors(evaluator, split='test'):
    """Categorize detection errors"""
    errors = {
        'classification': [],
        'localization': [],
        'duplicate': [],
        'background': [],
        'missed': []
    }
    return tide_results
```

**5. PR Curves** (~100 lines)
```python
def generate_pr_curves(evaluator, split='test'):
    """Generate precision-recall curves per class"""
    # Plot curves
    # Calculate AP
    # Save figures
    return pr_results
```

### Priority 3: Nice to have

**6. Visualizations** (~100 lines)
- GT vs Predictions side-by-side
- Confidence distribution plots
- Confusion matrices

---

## File Size Management

**Current:** 575 lines
**After additions:**
- Priority 1: ~925 lines
- Priority 2: ~1,175 lines
- Priority 3: ~1,275 lines

**Still manageable!** Compare to notebook's 3,171 lines.

---

## Recommended Next Steps

### Step 1: Add Priority 1 (CRITICAL) ✅
```bash
# Add to train_baseline.py:
1. prevalence_stratified_analysis()
2. per_class_analysis()
3. log_comprehensive_wandb()

# Test locally:
python train_baseline.py --model yolov8s --dataset d1 --epochs 2
```

### Step 2: Create Quick Mode Flag ✅
```python
parser.add_argument('--quick-mode', action='store_true',
                   help='Skip comprehensive evaluation (faster)')

if not args.quick_mode:
    # Run full evaluation
else:
    # Basic metrics only
```

### Step 3: Smoke Test ✅
```bash
# Test all combinations work
python smoke_test.py
# Tests: 2 models × 3 datasets × 1 epoch each
```

### Step 4: Cluster Submission Decision
```bash
# Option A: Full evaluation on cluster (slower but complete)
bash submit_experiments.sh  # No --quick-mode flag

# Option B: Training only on cluster (faster)
bash submit_experiments.sh --quick-mode
# Then run full evaluation locally on saved models
```

---

## W&B Dashboard Organization

After enhancement, each experiment will have:

```
W&B Run: yolov8s_d1_binary_no_weights_20251003_143022

├── Charts/
│   ├── training/
│   │   ├── box_loss, cls_loss, dfl_loss
│   │   └── learning_rate
│   ├── validation/
│   │   ├── mAP50, precision, recall, F1
│   │   └── infected/mAP50, uninfected/mAP50
│   └── test/
│       ├── mAP50, precision, recall, F1
│       └── infected/recall_by_density (CRITICAL!)
│
├── Tables/
│   ├── per_class_performance
│   ├── prevalence_stratified (CRITICAL!)
│   ├── tide_errors
│   ├── confusion_matrix
│   └── pr_curve_data
│
├── Images/
│   ├── pr_curves/
│   ├── visualizations/
│   └── sample_predictions/
│
└── Artifacts/
    ├── model_weights
    ├── evaluation_results.json
    └── analysis_data.csv
```

**All in ONE run - Complete story!** ✅

---

## Action Plan

**Immediate (Today):**
1. I'll enhance `train_baseline.py` with Priority 1 components
2. Add `--quick-mode` flag
3. Test locally with 2 epochs on D1

**Tomorrow:**
4. Create smoke test script
5. Validate on all 3 datasets
6. Verify W&B logging complete

**This Week:**
7. Submit to cluster with full evaluation
8. Monitor results in W&B

**This is the correct architecture!** ✅
