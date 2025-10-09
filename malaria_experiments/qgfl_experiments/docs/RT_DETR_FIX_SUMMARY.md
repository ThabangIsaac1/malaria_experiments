# RT-DETR Confidence Calibration Fix - Quick Summary

**Date:** 2025-10-06
**Status:** Ready to test 🚀

---

## Problem
RT-DETR baseline (200 epochs) produces **max confidence 0.33** → 0% test recall at conf=0.5

## Root Cause
**Using wrong optimizer!** SGD instead of AdamW due to iteration count threshold.

### Ultralytics Auto-Optimizer Logic
```python
name, lr = ("SGD", 0.01) if iterations > 10000 else ("AdamW", ~0.0017)
```

- **Our run:** 200 epochs × 150 batches = 30,000 iterations → **SGD selected** ❌
- **Guemas et al.:** 14 epochs × 31 batches = 438 iterations → **AdamW selected** ✓

## Solution
Force AdamW with appropriate hyperparameters for transformers.

---

## 3 Test Configurations (20 Epochs Each)

### Test 1: Standard AdamW
```bash
--optimizer AdamW --lr0 0.0017 --lrf 0.01 --warmup-epochs 5 --cls 1.0 --batch 16
```
**Goal:** Match Guemas et al. likely config

### Test 2: High Classification Weight
```bash
--optimizer AdamW --lr0 0.0017 --lrf 0.01 --warmup-epochs 5 --cls 2.0 --batch 16
```
**Goal:** Boost confidence scores explicitly

### Test 3: Faster Convergence
```bash
--optimizer AdamW --lr0 0.003 --lrf 0.01 --warmup-epochs 10 --cls 1.5 --batch 32
```
**Goal:** Match paper's batch size, faster learning

---

## Success Criteria
- ✅ **Primary:** Max confidence ≥ 0.5 (clinically usable)
- ✅ **Secondary:** Test recall @ conf=0.5 ≥ 50%
- ✅ **Tertiary:** Validation mAP50 ≥ 60%

---

## Files Modified
1. ✅ [cluster_run_baseline.py](../cluster_run_baseline.py) - Added CLI args for optimizer, lr, cls, box, warmup
2. ✅ [run_rtdetr_hyperparam_tests.sh](../run_rtdetr_hyperparam_tests.sh) - Test runner script
3. ✅ [RT_DETR_HYPERPARAMETER_ANALYSIS.md](RT_DETR_HYPERPARAMETER_ANALYSIS.md) - Detailed analysis

---

## How to Run Tests

```bash
cd /Users/thabangisaka/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/qgfl_experiments
./run_rtdetr_hyperparam_tests.sh
```

**Estimated time:** 6-8 hours (2-2.5 hours per test)

---

## Expected Outcome

Based on QGFL 20-epoch results (max conf 0.46 with wrong optimizer), we expect:

| Test | Max Conf (Predicted) | Reasoning |
|------|---------------------|-----------|
| Test 1 | 0.55-0.65 | Conservative, should work |
| Test 2 | 0.60-0.70 | High cls weight → higher confidence |
| Test 3 | 0.50-0.60 | Faster learning but may need more epochs |

**Best case:** Test 2 reaches conf ≥ 0.6 → deploy to cluster for 200 epochs

---

## Next Steps After Tests

1. Review W&B dashboard for max confidence scores
2. Check test recall @ conf=0.5 for each config
3. Select winning configuration
4. Deploy to cluster:
   - RT-DETR Baseline (corrected hyperparameters) × 4 datasets
   - RT-DETR QGFL (corrected hyperparameters) × 4 datasets
   - 200 epochs each

---

## Why This Matters

**User quote:** *"I can't use one architecture reviewers will ask about transformers so we need to solve this"*

- Need both CNN (YOLO) ✓ and Transformer (RT-DETR) ✗ for publication
- QGFL must work across architectures
- Can't deploy model with max conf=0.33 in clinical setting
- Fair comparison requires correct hyperparameters for both

---

## Reference
- **Paper:** Guemas et al. 2024 - RT-DETR for malaria detection
- **Key insight:** Short training (14 epochs) triggers AdamW, not SGD
- **Our mistake:** 200 epochs triggered wrong optimizer selection
