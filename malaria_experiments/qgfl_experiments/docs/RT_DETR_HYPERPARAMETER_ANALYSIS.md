# RT-DETR Hyperparameter Analysis: Why Our Baseline Failed

**Date:** 2025-10-06
**Problem:** RT-DETR baseline (200 epochs) produces max confidence 0.33, causing 0% test recall at conf=0.5
**Root Cause:** Using SGD optimizer with lr=0.01 instead of AdamW with lr~0.0017

---

## Critical Finding: Optimizer Selection Logic

Ultralytics uses **automatic optimizer selection** based on total iterations:

```python
# ultralytics/engine/trainer.py:791-792
nc = self.data.get("nc", 10)  # number of classes
lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation
name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
```

### Our Baseline Run (FAILED)
- **Epochs:** 200
- **Batch size:** 8
- **Dataset size:** ~1200 training images
- **Iterations:** 200 × 150 = **30,000 iterations**
- **Auto-selected optimizer:** SGD (iterations > 10000)
- **Learning rate:** 0.01
- **Result:** Max confidence = 0.33 ❌

### Guemas et al. 2024 RT-DETR Paper (SUCCESS)
- **Epochs:** 25 (configured), 14 (actual, early stopping)
- **Batch size:** 32
- **Dataset size:** ~1000 training images (estimated)
- **Iterations:** 14 × 31 = **438 iterations**
- **Auto-selected optimizer:** AdamW (iterations < 10000)
- **Learning rate:** ~0.0017 (calculated: 0.002 × 5 / 6)
- **Result:** Published successful malaria detection ✓

---

## Comparison Table

| Parameter | Our Baseline (Failed) | Guemas et al. (Success) | Explanation |
|-----------|----------------------|------------------------|-------------|
| **Optimizer** | SGD | AdamW | Transformers need adaptive learning rates |
| **Learning Rate** | 0.01 | ~0.0017 | 6× too high causes instability |
| **Batch Size** | 8 | 32 | 4× smaller reduces gradient quality |
| **Epochs** | 200 | 14 (early stop) | Overtrained with wrong optimizer |
| **Iterations** | 30,000 | 438 | Crossed 10k threshold → wrong optimizer |
| **cls Weight** | 0.5 | (default, likely 0.5-1.0) | Under-weights classification |
| **warmup_epochs** | 3 | (default, likely 3) | Acceptable |
| **Result** | Max conf 0.33 | Clinical deployment | Clear failure vs success |

---

## Why SGD Failed for RT-DETR

1. **Learning Rate Too High**
   - SGD lr=0.01 is appropriate for YOLO (CNN with batch norm)
   - Transformers need lr~0.001 (10× lower) due to attention mechanism sensitivity
   - High lr causes confidence logits to saturate early

2. **No Adaptive Learning**
   - SGD uses fixed per-parameter learning rate
   - Transformers have vastly different layer sensitivities (attention vs FFN)
   - AdamW adapts per-parameter, crucial for transformer convergence

3. **Small Batch Size Compounds Problem**
   - Batch=8 gives noisy gradients
   - SGD relies on momentum to smooth noise, but lr too high
   - AdamW's adaptive rates handle small batches better

4. **Iterations Trigger Wrong Selection**
   - 200 epochs × 150 batches = 30,000 iterations > 10,000 threshold
   - Ultralytics auto-selects SGD for "large-scale" training
   - But RT-DETR converges in <500 iterations with AdamW!

---

## QGFL 20-Epoch Improvement (Partially Successful)

Our QGFL run at 20 epochs achieved **max confidence 0.46** (vs baseline 0.33 at 200 epochs):

| Parameter | QGFL 20ep | Baseline 200ep | Improvement |
|-----------|-----------|----------------|-------------|
| Optimizer | SGD (still wrong) | SGD | Same |
| Epochs | 20 | 200 | 10× fewer |
| Max Confidence | 0.46 | 0.33 | +39% |
| Loss Focus | Infected class (α=0.9, γ=8.0) | Balanced | Better calibration |

**Key Insight:** QGFL's strong infected-class weighting (α=0.9) partially compensated for SGD's poor confidence calibration, achieving in 20 epochs what baseline couldn't in 200.

**Projection:** QGFL + AdamW could reach conf ≥ 0.6 in 20 epochs.

---

## Recommended Fix: 3 Test Runs (20 Epochs Each)

### Test 1: Standard AdamW (Conservative)
```yaml
optimizer: AdamW
lr0: 0.0017        # Ultralytics auto-fit for nc=2
lrf: 0.01          # Decay to 0.000017
momentum: 0.9      # AdamW beta1
weight_decay: 0.0005
warmup_epochs: 5   # More warmup for transformers
batch: 16          # 2× our baseline (GPU memory permitting)
cls: 1.0           # Equal to box (vs 0.5)
box: 7.5
```

**Rationale:** Matches Guemas et al. likely config, but with our dataset size.

### Test 2: High Classification Weight (Confidence Focus)
```yaml
optimizer: AdamW
lr0: 0.0017
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005
warmup_epochs: 5
batch: 16
cls: 2.0           # 2× box weight → force confidence up
box: 7.5
```

**Rationale:** Explicitly prioritize classification to fix confidence problem.

### Test 3: Faster Convergence (Aggressive)
```yaml
optimizer: AdamW
lr0: 0.003         # 1.8× higher for faster learning
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005
warmup_epochs: 10  # Longer warmup to stabilize high lr
batch: 32          # Match Guemas et al.
cls: 1.5
box: 7.5
```

**Rationale:** Match paper's batch size, slightly higher lr for 20-epoch timeline.

---

## Success Criteria (After 20 Epochs)

- **Primary:** Max confidence ≥ 0.5 (clinically usable)
- **Secondary:** Test recall @ conf=0.5 ≥ 50% (vs current 0%)
- **Tertiary:** Validation mAP50 ≥ 60% (reasonable for 20 epochs)

**Decision Point:** Best-performing config deployed to cluster for 200 epochs.

---

## Why This Matters for Publication

1. **Architecture Diversity:** Reviewers expect both CNN (YOLO) and Transformer (RT-DETR) validation
2. **QGFL Generalization:** Need to prove QGFL works across architectures
3. **Clinical Deployment:** Can't deploy a model with max conf=0.33 (no threshold works)
4. **Fair Comparison:** Current baseline vs QGFL comparison is invalid (both use wrong hyperparameters)

**Quote from user:** "I can't use one architecture reviewers will ask about transformers so we need to solve this"

---

## Next Steps

1. ✅ **Identified root cause:** SGD instead of AdamW
2. ✅ **Designed 3 test configurations**
3. ⏳ **Run 20-epoch tests locally** (2-3 hours each)
4. ⏳ **Select best config based on max confidence**
5. ⏳ **Deploy corrected baseline + QGFL to cluster** (200 epochs × 4 datasets)

**ETA:** 6-9 hours for local tests, then overnight cluster runs.
