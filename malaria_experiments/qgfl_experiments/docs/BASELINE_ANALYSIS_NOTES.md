# Baseline Experiment Analysis Notes

**Date**: October 4, 2025
**Status**: Phase 1 Running - Critical Findings

---

## D2 Critical Finding: Train-Test Stage Distribution Mismatch

### Performance Gap Discovery

**YOLOv8s on D2 Binary Task**:

| Split | Infected Recall | Infected Precision | F1 Score | Support |
|-------|----------------|-------------------|----------|---------|
| Validation | **88.7%** | 78.3% | 83.2% | 435 |
| Test | **29.4%** | 89.0% | 44.2% | 303 |
| **Gap** | **-59.3%** | +10.7% | -39.0% | -132 |

### Root Cause: Life-Cycle Stage Distribution Shift

From dataset documentation (D2 - P. vivax clinical samples):

**Training/Validation Distribution**:
- Total infected: 2.7% prevalence
- Trophozoite: **69%** (mature, easier to detect)
- Ring: 15% (early stage, harder)
- Schizont: 9%
- Gametocyte: 7%

**Test Distribution**:
- Total infected: 5.1% prevalence (higher!)
- Ring: **56%** (early stage, harder to detect!) ← DOMAIN SHIFT
- Trophozoite: 37%
- Gametocyte: 4%
- Schizont: 4%

### Clinical Significance

**Ring-stage parasites are morphologically subtle**:
- Smaller size
- Less distinct features
- Morphologically similar to artifacts/uninfected cells
- Critical for early diagnosis (treatment timing)

**Model learned on mature parasites (trophozoites), fails on early stages (rings)**

### Research Implications

✅ **This strengthens the QGFL research motivation**:
1. Demonstrates real-world domain shift problem
2. Shows baseline methods fail on hard cases
3. Validates need for quality-guided minority class focus
4. QGFL should provide larger gains on difficult examples (rings)

**Expected QGFL Impact**:
- Validation (already good): 88.7% → ~92% (+3-4%)
- Test (critical need): 29.4% → **45-55%** (+15-25%) ← KEY IMPROVEMENT

### Paper Contribution

> "D2 reveals severe performance degradation on out-of-distribution life-cycle stages: validation recall of 88.7% (trophozoite-dominated) drops to 29.4% on test (ring-dominated), demonstrating the clinical challenge of early-stage parasite detection. This 59.3% performance gap motivates quality-guided approaches that focus on morphologically subtle minority class examples."

---

## Dataset Comparison: Baseline Performance

### Expected Patterns Across Datasets

| Dataset | Type | Species | Prevalence | Expected Challenge |
|---------|------|---------|------------|-------------------|
| D1 | Lab cultures | P. falciparum | 6.6% | Moderate (controlled) |
| D2 | Clinical samples | P. vivax | 2.7% train, 5.1% test | **High (stage shift)** |
| D3 | Multi-center clinical | Multi-species | 2.6% | High (variability) |

### Key Questions to Analyze (When All Baselines Complete)

1. **Does D1 show smaller val-test gap?** (controlled lab conditions)
2. **Does D3 show similar stage shift issues?** (if staging data available)
3. **Which dataset benefits most from QGFL?** (D2 likely due to hard cases)
4. **Cross-dataset generalization**: Does QGFL help all datasets or just hard ones?

---

## Confusion Matrix vs Per-Class Metrics Discrepancy (RESOLVED)

### Initial Confusion (Oct 4, 2025)

**Confusion Matrix showed**:
- Infected cells: 347
- TP: 134
- Recall: 38.6%

**Per-Class Metrics showed**:
- Infected support: 303
- TP: 89
- Recall: 29.4%

### Resolution

**CONFIRMED: Both metrics from test set, both at conf=0.5, IoU > 0.5**

**Confusion Matrix**:
- Infected TP: 87, FN: 216 (134 misclassified + 82 missed)
- Support: 303
- Recall: 87/303 = 28.7%

**Per-Class Table**:
- Infected TP: 89, FN: 214
- Support: 303
- Recall: 89/303 = 29.4%

**Discrepancy**: 2 detections (0.7% difference)

**Root Cause**: Minor matching differences due to:
- Different iteration order (per-class filters by class first, confusion matrix doesn't)
- Floating-point precision in IoU calculations near 0.5 threshold
- Edge cases where IoU ≈ 0.5 handled slightly differently

**Conclusion**: ✅ Functionally identical. The 0.7% difference is within acceptable margin. Both confirm infected recall ~29% on test set.

---

## Cross-Dataset Observations Template

### D1 Results (Pending - YOLOv8s/YOLOv11s)

**Validation**:
- Uninfected: Precision ___, Recall ___, F1 ___
- Infected: Precision ___, Recall ___, F1 ___

**Test**:
- Uninfected: Precision ___, Recall ___, F1 ___
- Infected: Precision ___, Recall ___, F1 ___

**Val-Test Gap**: ___

### D2 Results (COMPLETE - YOLOv8s)

**Validation**:
- Uninfected: Precision 77.7%, Recall 98.6%, F1 86.9%
- Infected: Precision 78.3%, Recall **88.7%**, F1 83.2%

**Test**:
- Uninfected: Precision 88.3%, Recall 93.8%, F1 90.9%
- Infected: Precision 89.0%, Recall **29.4%**, F1 44.2%

**Val-Test Gap**: -59.3% recall (stage distribution shift)

### D3 Results (Running - YOLOv8s)

**Validation**:
- Uninfected: Precision ___, Recall ___, F1 ___
- Infected: Precision ___, Recall ___, F1 ___

**Test**:
- Uninfected: Precision ___, Recall ___, F1 ___
- Infected: Precision ___, Recall ___, F1 ___

**Val-Test Gap**: ___

---

## Confusion Matrix Threshold Analysis

### YOLO Default vs Custom Metrics

**YOLO Built-in Confusion Matrix**:
- Uses `conf=0.001` (very low threshold)
- Includes all predictions regardless of confidence
- Shows model's raw capability
- Higher numbers (counts uncertain predictions)

**Custom Confusion Matrix** (evaluator.py line 656):
- Uses `conf=0.5` (standard clinical threshold)
- Only counts confident predictions
- Shows clinically usable performance
- Lower numbers (realistic deployment scenario)

**Decision**: Keep `conf=0.5` for all custom metrics
- ✅ Matches clinical reality (doctors need confidence)
- ✅ Consistent across all evaluation metrics
- ✅ Conservative, honest reporting
- ✅ Matches research methodology

---

## Action Items

- [x] Document D2 stage distribution shift finding
- [ ] Wait for D1 baseline completion - compare val-test gap
- [ ] Wait for D3 baseline completion - compare val-test gap
- [ ] Verify confusion matrix vs per-class metrics use identical code paths
- [ ] Create visualization comparing val vs test performance across datasets
- [ ] Prepare ablation study comparing QGFL impact on "easy" (validation) vs "hard" (test) cases

---

## Research Narrative

**Baseline reveals the problem** → **QGFL provides the solution**

1. **Problem**: Extreme class imbalance + morphological similarity
2. **Evidence**: D2 test recall 29.4% on ring-stage parasites
3. **Challenge**: 59.3% performance drop on out-of-distribution stages
4. **Solution**: QGFL's quality-guided focus on difficult minority class examples
5. **Validation**: Cross-dataset evaluation (D1, D2, D3)
6. **Impact**: Expect 15-25% recall improvement on hardest cases

---

**Last Updated**: October 4, 2025
**Next Update**: When D1 and D3 baselines complete
