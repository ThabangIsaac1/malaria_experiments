# QGFL Quick Reference Guide

**Last Updated:** 2025-10-05
**Purpose:** Fast lookup for QGFL parameters and architecture integration points

---

## Architecture Integration Points

### YOLO (YOLOv8s, YOLOv11s)

| Aspect | Value |
|--------|-------|
| **Loss Class** | `v8DetectionLoss` |
| **File** | `ultralytics/utils/loss.py` |
| **Integration Line** | Line 247: `loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum` |
| **Current Loss** | Standard BCE (NO focal, NO class weighting) |
| **Target Format** | Soft targets ∈ [0, 1] (IoU-quality weighted) |
| **Num Predictions** | ~8400 per image |
| **Assignment** | Task-Aligned Assigner (soft) |

**Replace:** Line 247 BCE → QGFL classification loss

---

### RT-DETR

| Aspect | Value |
|--------|-------|
| **Loss Class** | `RTDETRDetectionLoss` (extends `DETRLoss`) |
| **File** | `ultralytics/models/utils/loss.py` |
| **Integration Lines** | Lines 77-84 in `_get_loss_class()` method |
| **Current Loss** | **VarifocalLoss** (α=0.75, γ=2.0) - NOT BCE! |
| **Target Format** | One-hot + IoU quality scores (separate) |
| **Num Predictions** | 300 queries per image |
| **Assignment** | Hungarian Matcher (bipartite 1:1) |

**Replace:** Lines 77-84 VarifocalLoss → QGFL classification loss

⚠️ **IMPORTANT:** RT-DETR baseline already uses focal loss. QGFL must beat VarifocalLoss, not just BCE!

---

## QGFL Parameters (From Paper)

### Class Weights (α)

```python
INFECTED_ALPHA = 0.9      # Focus 90% on infected (minority class)
UNINFECTED_ALPHA = 0.1    # Focus 10% on uninfected (majority class)
```

**Why:** Addresses severe class imbalance (0.7%-2.8% infected across datasets)

---

### Focusing Power (γ)

```python
INFECTED_GAMMA = 8.0      # Strong focus on hard infected examples
UNINFECTED_GAMMA = 4.0    # Moderate focus on hard uninfected examples
```

**Why:** Infected class (minority) needs 2× stronger focusing to overcome gradient dominance by majority class

⚠️ **Watch for:** May need reduction for RT-DETR (300 queries vs RetinaNet's 100k anchors)

---

### Difficulty Threshold

```python
DIFFICULTY_THRESHOLD = 0.925
```

**What it does:** Only apply QGFL to predictions with confidence < 0.925 (pt < 0.925)

**Formula:**
```python
difficulty = max(raw_difficulty - 0.925, 0) / (1 - 0.925)
raw_difficulty = 1 - pt
```

**Why:** Prevents trivial easy examples from adding noise. Focus only on genuinely challenging samples.

---

### Quality Components

```python
QUALITY_MARGIN = 0.5      # Ignore quality differences < 0.5
QUALITY_FACTOR = 2.0      # Exponential scaling for quality weighting
```

**Formula:**
```python
quality = |prediction - target|
quality_adjusted = max(quality - 0.5, 0)
quality_weight = min(quality_adjusted^2.0, 10.0)
```

**Why:** Amplifies loss for low-quality predictions (large |p - target|), reduces for high-quality.

---

### UIoU Decay

```python
UIOU_START = 2.0          # Initial emphasis (localization focus)
UIOU_END = 0.5            # Final emphasis (classification focus)
```

**Formula:**
```python
progress = current_epoch / total_epochs
uiou_ratio = 2.0 + (0.5 - 2.0) × progress
```

**Why:** Early training focuses on localization (high UIoU), later on classification (low UIoU).

⚠️ **Watch for:** May conflict with YOLO's DFL (Distribution Focal Loss). Consider ablation with UIoU=1.0 (disabled).

---

## Complete QGFL Formula

```python
QGFL(pt) = -αt × (1 - pt)^γeff × (1 + quality_weight) × uiou_ratio × log(pt)

Where:
  pt = predicted probability for true class
  αt = class-specific alpha (0.9 for infected, 0.1 for uninfected)
  γeff = base_γ + (max_γ - base_γ) × difficulty
       = 2.0 + (8.0 - 2.0) × difficulty (infected)
       = 2.0 + (4.0 - 2.0) × difficulty (uninfected)
  difficulty = max((1-pt) - 0.925, 0) / (1 - 0.925)
  quality_weight = min((max(|p-target| - 0.5, 0))^2.0, 10.0)
  uiou_ratio = 2.0 → 0.5 (linear decay)
```

---

## Expected Improvements (From Paper)

### Recall @ 1-3% Parasitemia (Critical Range)

| Dataset | Baseline | QGFL | Improvement |
|---------|----------|------|-------------|
| D1 (P. falciparum) | 0.42 | 0.61 | **+46%** |
| D2 (P. vivax) | 0.28 | 0.54 | **+93%** |
| D3 (Mixed) | 0.71 | 0.76 | **+8%** |

### Overall Metrics

| Metric | D1 | D2 | D3 |
|--------|----|----|---- |
| **mAP Improvement** | +5.7% | +11.2% | +10.4% |
| **Infected F1 Improvement** | +16.5% | +14.7% | - |
| **Missed Detection Reduction** | 52.5%→38.1% | - | - |

---

## Implementation Checklist

### Phase 1: Core Implementation

- [ ] `src/losses/qgfl_core.py` - Shared QGFL components
  - [ ] `compute_gamma_eff()` - Difficulty-aware + class-specific γ
  - [ ] `compute_quality_weight()` - Quality-guided weighting
  - [ ] `compute_uiou_ratio()` - UIoU decay schedule

- [ ] `src/losses/qgfl_yolo.py` - YOLO-specific
  - [ ] Handle soft targets (IoU-weighted)
  - [ ] Integrate with Task-Aligned Assigner outputs
  - [ ] Normalization: sum() / target_scores_sum

- [ ] `src/losses/qgfl_rtdetr.py` - RT-DETR-specific
  - [ ] Handle one-hot + quality scores
  - [ ] Integrate with Hungarian Matcher outputs
  - [ ] Normalization: mean(1).sum() / max(num_gts, 1) * nq

### Phase 2: Integration

- [ ] Create `cluster_run_qgfl.py` (copy of cluster_run_baseline.py)
- [ ] Add `--loss-type qgfl` argument
- [ ] Implement monkey-patching for YOLO loss
- [ ] Implement monkey-patching for RT-DETR loss
- [ ] Add QGFL parameters to config

### Phase 3: Testing

- [ ] **Stability Test** - 5 epochs, batch=4
  - [ ] YOLO: Loss converges, no NaN/Inf
  - [ ] RT-DETR: Loss converges, no NaN/Inf
  - [ ] Both: mAP > 0.3 (shows learning)

- [ ] **Full Experiments** - 200 epochs, batch=16
  - [ ] 9 baseline (already running on cluster)
  - [ ] 9 QGFL (pending stability test)

---

## Monitoring During Training

### Red Flags (Stop & Debug)

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss < 0.001 (epoch 1) | Gamma too high | Reduce γ: 8.0→6.0, 4.0→3.0 |
| Loss > 100 (epoch 1) | Normalization issue | Check target_scores_sum |
| NaN/Inf | Numerical instability | Add epsilon to log, check pt bounds |
| mAP < 0.1 (epoch 5) | Model not learning | Check gradient flow, loss components |
| Gradients vanish | Over-suppression | Reduce γ or threshold |

### Good Signs

| Metric | Expected Behavior |
|--------|-------------------|
| **Loss** | Smooth decay from ~2.0 → ~0.5 over 200 epochs |
| **Infected Recall** | Steady increase, especially @ 1-3% density |
| **Uninfected Metrics** | Stable (don't degrade vs baseline) |
| **Gradients** | Non-zero, stable magnitude |

---

## Parameter Tuning (If Stability Fails)

### For YOLO

**If loss collapses:**
```python
INFECTED_GAMMA = 6.0  # Reduce from 8.0
UNINFECTED_GAMMA = 3.0  # Reduce from 4.0
```

**If DFL conflict:**
```python
UIOU_START = 1.0  # Disable decay
UIOU_END = 1.0
```

### For RT-DETR

**If loss collapses (300 queries vs 8400):**
```python
INFECTED_GAMMA = 6.0  # Reduce from 8.0
UNINFECTED_GAMMA = 3.0  # Reduce from 4.0
```

**Keep UIoU decay** (no DFL in RT-DETR):
```python
UIOU_START = 2.0
UIOU_END = 0.5
```

---

## File Structure

```
qgfl_experiments/
├── src/
│   └── losses/
│       ├── __init__.py
│       ├── qgfl_core.py          # Shared QGFL logic
│       ├── qgfl_yolo.py          # YOLO-specific implementation
│       ├── qgfl_rtdetr.py        # RT-DETR-specific implementation
│       └── baseline/              # Future: other loss variants
│           └── focal_loss.py     # Standard focal loss (for comparison)
├── cluster_run_baseline.py       # Baseline experiments (BCE/VarifocalLoss)
├── cluster_run_qgfl.py           # QGFL experiments
└── docs/
    ├── QGFL_QUICK_REFERENCE.md   # This file
    └── QGFL_ARCHITECTURE_ANALYSIS.md  # Detailed analysis
```

---

## Future Extensions (Modular Design)

### Multi-Species Classification (D3)

**Current:** Binary (Infected vs Uninfected)
**Future:** 4-class (Uninfected, P. falciparum, P. vivax, P. malariae)

**QGFL Extension:**
```python
# Multi-species class weights (based on prevalence)
CLASS_ALPHAS = {
    'uninfected': 0.05,      # Majority (95%)
    'falciparum': 0.40,      # Most common parasite
    'vivax': 0.35,           # Second common
    'malariae': 0.20         # Rare
}

# Multi-species gammas
CLASS_GAMMAS = {
    'uninfected': 2.0,
    'falciparum': 6.0,
    'vivax': 6.0,
    'malariae': 8.0         # Rarest gets strongest focus
}
```

**Implementation:**
- Extend `qgfl_core.py` to support multi-class
- Create `src/losses/qgfl_multiclass.py`
- Use same core components, different class indexing

---

### Staging Classification (D1, D2)

**Goal:** Detect parasite life stage (ring, trophozoite, schizont, gametocyte)

**QGFL Extension:**
```python
# Hierarchical loss: species → stage
# Stage-specific alphas based on frequency
STAGE_ALPHAS = {
    'ring': 0.4,           # Most common
    'trophozoite': 0.3,
    'schizont': 0.2,
    'gametocyte': 0.1      # Rarest
}
```

**Implementation:**
- Create `src/losses/qgfl_hierarchical.py`
- Two-stage QGFL: coarse (infected/not) + fine (stage)
- Weighted combination of both losses

---

### DN-DETR (Denoising Training for RT-DETR)

**Current:** RT-DETR uses standard denoising (dn_loss)
**Future:** Apply QGFL to denoising branch as well

**QGFL Extension:**
```python
# In qgfl_rtdetr.py
def forward_with_denoising(self, preds, batch, dn_bboxes, dn_scores, dn_meta):
    # Main detection loss (already QGFL)
    main_loss = self.forward(preds, batch)

    # Denoising loss (apply QGFL here too)
    if dn_meta is not None:
        dn_loss = self.forward(dn_bboxes, dn_scores, batch, postfix='_dn', ...)
        main_loss.update(dn_loss)

    return main_loss
```

**Implementation:**
- Extend `qgfl_rtdetr.py` to handle dn_meta
- Apply same QGFL principles to denoising branch
- Consistent class/difficulty weighting

---

## Modular Design Principles

### 1. Separation of Concerns

```
qgfl_core.py       → Architecture-agnostic QGFL math
qgfl_yolo.py       → YOLO-specific integration
qgfl_rtdetr.py     → RT-DETR-specific integration
qgfl_multiclass.py → Multi-class extension (future)
qgfl_hierarchical.py → Staging extension (future)
```

### 2. Configuration Management

```python
# configs/qgfl_config.py
QGFL_CONFIGS = {
    'binary': {
        'infected_alpha': 0.9,
        'uninfected_alpha': 0.1,
        ...
    },
    'multispecies': {
        'class_alphas': {...},
        'class_gammas': {...},
        ...
    },
    'staging': {
        'stage_alphas': {...},
        'hierarchical': True,
        ...
    }
}
```

### 3. Loss Registry

```python
# src/losses/__init__.py
LOSS_REGISTRY = {
    'baseline': None,  # Use architecture default
    'qgfl_binary': (QGFLYOLOLoss, QGFLRTDETRLoss),
    'qgfl_multiclass': (QGFLMultiClassYOLO, QGFLMultiClassRTDETR),
    'qgfl_hierarchical': (QGFLHierarchicalYOLO, QGFLHierarchicalRTDETR),
}

def get_loss(loss_type, architecture, **kwargs):
    if architecture == 'yolo':
        return LOSS_REGISTRY[loss_type][0](**kwargs)
    elif architecture == 'rtdetr':
        return LOSS_REGISTRY[loss_type][1](**kwargs)
```

### 4. Training Script Modularity

```
cluster_run_baseline.py   → Baseline (BCE/VarifocalLoss)
cluster_run_qgfl.py       → QGFL binary
cluster_run_multiclass.py → QGFL multi-species (future)
cluster_run_staging.py    → QGFL staging (future)
```

Each script:
- Shares common code via imports
- Has specific loss configuration
- Logs to separate W&B projects

---

## Quick Start Commands

### Stability Test

```bash
# YOLO
python cluster_run_qgfl.py --model yolov8s --dataset d1 --task binary \
    --epochs 5 --batch-size 4 --loss-type qgfl

# RT-DETR
python cluster_run_qgfl.py --model rtdetr-l --dataset d1 --task binary \
    --epochs 5 --batch-size 4 --loss-type qgfl
```

### Full Experiment

```bash
# Submit all 9 QGFL experiments to cluster
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts

sbatch run_d1_yolov8s_qgfl.sh
sbatch run_d1_yolov11s_qgfl.sh
sbatch run_d1_rtdetr_qgfl.sh
sbatch run_d2_yolov8s_qgfl.sh
sbatch run_d2_yolov11s_qgfl.sh
sbatch run_d2_rtdetr_qgfl.sh
sbatch run_d3_yolov8s_qgfl.sh
sbatch run_d3_yolov11s_qgfl.sh
sbatch run_d3_rtdetr_qgfl.sh
```

---

## References

- **QGFL Paper:** Quality-Guided Focal Loss: Enhancing Minority Class Detection in Haematological Imaging
- **Focal Loss:** Lin et al., "Focal Loss for Dense Object Detection" (2018)
- **VarifocalLoss:** Zhang et al., "VarifocalNet" (2020)
- **RT-DETR:** Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection" (2023)
- **YOLO:** Ultralytics YOLOv8/v11 Documentation

---

**Last Updated:** 2025-10-05
**Status:** Ready for implementation
