# Future Extensions & Modularity Plan

**Date:** 2025-10-05
**Purpose:** Strategic roadmap for extending QGFL to multi-species, staging, and DN-DETR

---

## Current State (Phase 1: Binary Classification)

### Scope
- **Task:** Binary classification (Infected vs Uninfected)
- **Datasets:** D1, D2, D3
- **Models:** YOLOv8s, YOLOv11s, RT-DETR-L
- **Loss:** QGFL adapted for binary classification

### Architecture
```
src/losses/
├── qgfl_core.py          # Shared QGFL components (binary)
├── qgfl_yolo.py          # YOLO-specific binary QGFL
└── qgfl_rtdetr.py        # RT-DETR-specific binary QGFL
```

---

## Phase 2: Multi-Species Classification (D3)

### Objective
Classify parasite species: Uninfected, P. falciparum, P. vivax, P. malariae

### Dataset Modification (D3 Only)
**Current D3 Structure:**
```yaml
# d3_binary.yaml
classes:
  0: Uninfected
  1: Infected
nc: 2
```

**Future D3 Structure:**
```yaml
# d3_multispecies.yaml
classes:
  0: Uninfected
  1: P. falciparum
  2: P. vivax
  3: P. malariae
nc: 4
```

### QGFL Extension Strategy

#### Multi-Class Alpha Weighting

**Based on D3 prevalence:**
```python
# configs/qgfl_multiclass_config.py
MULTISPECIES_ALPHAS = {
    0: 0.05,   # Uninfected (95% - majority, lowest weight)
    1: 0.40,   # P. falciparum (most common parasite)
    2: 0.35,   # P. vivax (second most common)
    3: 0.20    # P. malariae (rarest, highest weight among parasites)
}

MULTISPECIES_GAMMAS = {
    0: 2.0,    # Uninfected (easy to detect)
    1: 6.0,    # P. falciparum (moderate difficulty)
    2: 6.0,    # P. vivax (moderate difficulty)
    3: 8.0     # P. malariae (rarest, hardest to detect)
}
```

#### Implementation

**New File:** `src/losses/qgfl_multiclass.py`

```python
class QGFLMultiClass(QGFLCore):
    """Multi-class extension of QGFL"""
    def __init__(self, nc, class_alphas, class_gammas, **qgfl_params):
        super().__init__(**qgfl_params)
        self.nc = nc
        self.class_alphas = class_alphas  # Dict: {class_id: alpha}
        self.class_gammas = class_gammas  # Dict: {class_id: gamma}

    def get_class_alpha(self, class_ids):
        """Map class IDs to their alphas"""
        return torch.tensor([self.class_alphas[c.item()] for c in class_ids])

    def get_class_gamma(self, class_ids):
        """Map class IDs to their gammas"""
        return torch.tensor([self.class_gammas[c.item()] for c in class_ids])
```

**Training Script:** `cluster_run_multiclass.py`

```bash
python cluster_run_multiclass.py --model yolov8s --dataset d3 \
    --task multispecies --epochs 200 --batch-size 16 --loss-type qgfl_multiclass
```

---

## Phase 3: Staging Classification (D1, D2)

### Objective
Classify parasite life stage: Ring, Trophozoite, Schizont, Gametocyte

### Dataset Modification (D1, D2)

**Current Structure:**
```yaml
# d1_binary.yaml
classes:
  0: Uninfected
  1: Infected
```

**Future Structure (Option 1 - Flat):**
```yaml
# d1_staging.yaml
classes:
  0: Uninfected
  1: Ring
  2: Trophozoite
  3: Schizont
  4: Gametocyte
nc: 5
```

**Future Structure (Option 2 - Hierarchical):**
```yaml
# d1_hierarchical.yaml
hierarchy:
  level_1:  # Coarse
    0: Uninfected
    1: Infected
  level_2:  # Fine (only for infected)
    0: Ring
    1: Trophozoite
    2: Schizont
    3: Gametocyte
```

### QGFL Extension Strategy

#### Option 1: Flat Multi-Class (Simpler)

```python
# configs/qgfl_staging_config.py
STAGING_ALPHAS = {
    0: 0.05,   # Uninfected (majority)
    1: 0.35,   # Ring (most common stage)
    2: 0.30,   # Trophozoite
    3: 0.20,   # Schizont
    4: 0.10    # Gametocyte (rarest)
}

STAGING_GAMMAS = {
    0: 2.0,    # Uninfected
    1: 6.0,    # Ring
    2: 6.0,    # Trophozoite
    3: 7.0,    # Schizont (harder to detect)
    4: 8.0     # Gametocyte (rarest, hardest)
}
```

Use `QGFLMultiClass` (same as multi-species)

#### Option 2: Hierarchical (More Complex, Better Performance)

```python
class QGFLHierarchical(nn.Module):
    """Two-stage QGFL: coarse (infected/not) + fine (stage)"""
    def __init__(self, nc_coarse, nc_fine, **qgfl_params):
        self.qgfl_coarse = QGFLMultiClass(nc=nc_coarse, ...)  # Binary
        self.qgfl_fine = QGFLMultiClass(nc=nc_fine, ...)      # 4-class staging

    def forward(self, pred_scores, targets, ...):
        # Coarse loss (infected vs uninfected)
        loss_coarse = self.qgfl_coarse(pred_scores_coarse, targets_coarse, ...)

        # Fine loss (stage classification - only for infected)
        infected_mask = (targets_coarse == 1)
        loss_fine = self.qgfl_fine(pred_scores_fine[infected_mask],
                                    targets_fine[infected_mask], ...)

        # Weighted combination
        total_loss = 0.6 * loss_coarse + 0.4 * loss_fine
        return total_loss
```

**Implementation:** `src/losses/qgfl_hierarchical.py`

**Training Script:** `cluster_run_staging.py`

---

## Phase 4: DN-DETR Integration (RT-DETR Only)

### Objective
Apply QGFL to RT-DETR's denoising training branch

### Background: What is DN-DETR?

RT-DETR uses **denoising training** to improve convergence:
1. **Main branch:** Standard detection queries (300)
2. **Denoising branch:** Noisy GT queries + reconstruction task
3. **Loss:** Main detection loss + denoising reconstruction loss

**Current:** Denoising uses VarifocalLoss (same as main branch)
**Goal:** Apply QGFL to denoising branch for consistency

### QGFL Extension Strategy

**Modify:** `src/losses/qgfl_rtdetr.py`

```python
class QGFLRTDETRLoss(QGFLCore):
    def forward(self, pred_scores, one_hot, gt_scores, targets,
                num_gts, nq, current_epoch=0, total_epochs=200,
                is_denoising=False):  # NEW PARAMETER
        """
        Compute QGFL loss for RT-DETR

        Args:
            is_denoising: If True, apply to denoising branch
        """
        # Same QGFL logic for both main and denoising branches
        # Ensures consistent class/difficulty weighting

        # ... (existing QGFL computation) ...

        if is_denoising:
            # Denoising-specific normalization (if needed)
            pass

        return loss

class QGFLRTDETRWithDenoising(QGFLRTDETRLoss):
    """RT-DETR QGFL with denoising support"""
    def forward_with_denoising(self, preds, batch, dn_bboxes=None,
                               dn_scores=None, dn_meta=None):
        # Main detection loss
        main_loss = super().forward(preds, batch, is_denoising=False)

        # Denoising loss (if dn_meta provided)
        if dn_meta is not None:
            dn_loss = super().forward(dn_bboxes, dn_scores, batch,
                                      is_denoising=True, dn_meta=dn_meta)
            main_loss.update({'loss_class_dn': dn_loss})

        return main_loss
```

**Training:** No new script needed - automatically used when RT-DETR model provides `dn_meta`

---

## Modular Architecture Design

### Loss Module Structure

```
src/losses/
├── __init__.py              # Loss registry
├── qgfl_core.py             # Core QGFL math (architecture-agnostic)
│
├── Binary Classification (Phase 1 - CURRENT)
│   ├── qgfl_yolo.py         # YOLO binary QGFL
│   └── qgfl_rtdetr.py       # RT-DETR binary QGFL
│
├── Multi-Species (Phase 2 - D3)
│   ├── qgfl_multiclass.py   # Multi-class QGFL core
│   ├── qgfl_multiclass_yolo.py
│   └── qgfl_multiclass_rtdetr.py
│
├── Staging (Phase 3 - D1, D2)
│   ├── qgfl_hierarchical.py # Hierarchical QGFL core
│   ├── qgfl_hierarchical_yolo.py
│   └── qgfl_hierarchical_rtdetr.py
│
└── Baseline Losses (For Comparison)
    ├── focal_loss.py        # Standard focal loss
    └── varifocal_loss.py    # VarifocalLoss
```

### Loss Registry (`src/losses/__init__.py`)

```python
from .qgfl_yolo import QGFLYOLOLoss
from .qgfl_rtdetr import QGFLRTDETRLoss
# Future imports...

LOSS_REGISTRY = {
    # Binary classification
    'baseline': None,  # Use architecture default
    'qgfl_binary': {
        'yolo': QGFLYOLOLoss,
        'rtdetr': QGFLRTDETRLoss
    },

    # Multi-species (Phase 2)
    'qgfl_multiclass': {
        'yolo': QGFLMultiClassYOLO,
        'rtdetr': QGFLMultiClassRTDETR
    },

    # Staging (Phase 3)
    'qgfl_hierarchical': {
        'yolo': QGFLHierarchicalYOLO,
        'rtdetr': QGFLHierarchicalRTDETR
    },

    # Comparisons
    'focal': {
        'yolo': FocalLossYOLO,
        'rtdetr': FocalLossRTDETR
    }
}

def get_loss(loss_type, architecture, **kwargs):
    """Get loss function by type and architecture"""
    if loss_type == 'baseline':
        return None  # Use default

    loss_dict = LOSS_REGISTRY.get(loss_type)
    if loss_dict is None:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_class = loss_dict.get(architecture)
    if loss_class is None:
        raise ValueError(f"Loss {loss_type} not implemented for {architecture}")

    return loss_class(**kwargs)
```

### Configuration Management

```
configs/
├── qgfl_binary_config.py       # Phase 1 (current)
├── qgfl_multiclass_config.py   # Phase 2
├── qgfl_hierarchical_config.py # Phase 3
└── baseline_config.py          # Existing baseline config
```

**Example: `configs/qgfl_multiclass_config.py`**

```python
# D3 Multi-Species Configuration
QGFL_MULTICLASS_PARAMS = {
    'nc': 4,
    'class_alphas': {0: 0.05, 1: 0.40, 2: 0.35, 3: 0.20},
    'class_gammas': {0: 2.0, 1: 6.0, 2: 6.0, 3: 8.0},
    'difficulty_threshold': 0.925,
    'quality_margin': 0.5,
    'quality_factor': 2.0,
    'uiou_start': 2.0,
    'uiou_end': 0.5
}
```

### Training Scripts

```
qgfl_experiments/
├── cluster_run_baseline.py     # Baseline (BCE/VarifocalLoss)
├── cluster_run_qgfl.py         # Phase 1: Binary QGFL
├── cluster_run_multiclass.py   # Phase 2: Multi-species QGFL
└── cluster_run_staging.py      # Phase 3: Hierarchical QGFL
```

**Shared code via:** `src/training_utils.py`

```python
# src/training_utils.py
def setup_model_and_loss(config):
    """Shared setup logic for all training scripts"""
    # Load model
    if config.model_name in ['yolov8s', 'yolov11s']:
        model = YOLO(f'{config.model_name}.pt')
        architecture = 'yolo'
    elif config.model_name == 'rtdetr-l':
        model = RTDETR('rtdetr-l.pt')
        architecture = 'rtdetr'

    # Get loss function
    if config.loss_type != 'baseline':
        loss_fn = get_loss(
            loss_type=config.loss_type,
            architecture=architecture,
            **config.loss_params
        )
        # Apply monkey-patch
        patch_model_loss(model, loss_fn, architecture)

    return model
```

---

## Experimental Timeline

### Phase 1: Binary QGFL (Current - Week 1-2)

**Tasks:**
- [x] Architecture analysis
- [ ] Implement QGFL core, YOLO, RT-DETR
- [ ] Stability test (5 epochs)
- [ ] Full experiments (200 epochs, 9 runs)
- [ ] Analysis & paper writing

**Deliverable:** QGFL binary classification results (D1, D2, D3)

---

### Phase 2: Multi-Species (Week 3-4)

**Prerequisites:**
- Phase 1 complete
- D3 dataset re-annotated for species-level labels

**Tasks:**
- [ ] Re-annotate D3 (Uninfected, P.f, P.v, P.m)
- [ ] Implement `qgfl_multiclass.py`
- [ ] Extend YOLO/RT-DETR wrappers
- [ ] Stability test
- [ ] Full experiments (3 models × D3 = 3 runs)
- [ ] Compare vs binary classification

**Deliverable:** Species-level detection results (D3)

---

### Phase 3: Staging (Week 5-6)

**Prerequisites:**
- Phase 2 complete (demonstrates multi-class works)
- D1/D2 datasets re-annotated for life stages

**Tasks:**
- [ ] Re-annotate D1/D2 (stages: Ring, Troph, Schiz, Gamet)
- [ ] Decide: Flat vs Hierarchical
- [ ] Implement `qgfl_hierarchical.py` (if hierarchical)
- [ ] Stability test
- [ ] Full experiments (3 models × 2 datasets = 6 runs)
- [ ] Clinical validation (stage accuracy)

**Deliverable:** Life stage classification results (D1, D2)

---

### Phase 4: DN-DETR (Week 7 - Optional)

**Prerequisites:**
- Phase 1-3 complete
- RT-DETR showing good results

**Tasks:**
- [ ] Extend `qgfl_rtdetr.py` for denoising
- [ ] Test denoising convergence improvement
- [ ] Ablation: QGFL main-only vs QGFL main+dn

**Deliverable:** DN-DETR ablation study

---

## W&B Project Organization

### Current (Phase 1)
```
Project: malaria_qgfl_experiments
Tags: binary, baseline, qgfl
Groups: yolov8s, yolov11s, rtdetr
```

### Future (Phase 2-4)
```
Project: malaria_qgfl_multispecies
Tags: multispecies, d3
Groups: yolov8s, yolov11s, rtdetr

Project: malaria_qgfl_staging
Tags: staging, hierarchical, d1, d2
Groups: yolov8s, yolov11s, rtdetr
```

---

## Modularity Benefits

### 1. Clean Separation
- Each phase has own loss module
- No breaking changes to previous phases
- Easy to compare (binary vs multi-species vs staging)

### 2. Code Reuse
- `qgfl_core.py` shared across all phases
- `QGFLMultiClass` used for both multi-species AND staging (flat)
- Training utils shared across all scripts

### 3. Experimentation Flexibility
```bash
# Binary QGFL (Phase 1)
python cluster_run_qgfl.py --loss-type qgfl_binary

# Multi-species (Phase 2)
python cluster_run_multiclass.py --loss-type qgfl_multiclass --dataset d3

# Staging (Phase 3)
python cluster_run_staging.py --loss-type qgfl_hierarchical --dataset d1

# Compare different loss functions
python cluster_run_qgfl.py --loss-type focal  # Standard focal loss baseline
```

### 4. Paper Writing
Each phase = separate paper section:
- **Section 4.1:** Binary QGFL (current)
- **Section 4.2:** Multi-species extension (D3)
- **Section 4.3:** Hierarchical staging (D1, D2)
- **Section 4.4:** DN-DETR ablation

---

## Decision Points

### Multi-Species: When to Extend?

**Option A:** After Phase 1 completes successfully
- Pro: Demonstrates full capability
- Con: More work, delays Phase 1 publication

**Option B:** Separate follow-up paper
- Pro: Faster Phase 1 publication
- Con: Less comprehensive

**Recommendation:** **Option A** - Include multi-species in same paper as strong evidence of generalizability

---

### Staging: Flat vs Hierarchical?

**Flat (5-class):**
- Pro: Simpler implementation (reuse `QGFLMultiClass`)
- Con: Treats uninfected and stages equally (misses hierarchy)

**Hierarchical (2-level):**
- Pro: Matches clinical workflow (first detect, then stage)
- Pro: Better class imbalance handling (infected first, then stages)
- Con: More complex implementation

**Recommendation:** **Hierarchical** - More clinically relevant, better handles extreme imbalance

---

### DN-DETR: Worth the effort?

**If RT-DETR baseline struggles:**
- YES - Denoising might help convergence

**If RT-DETR baseline works well:**
- MAYBE - Nice ablation study but not critical

**Recommendation:** Assess after Phase 1 RT-DETR results

---

## Summary

**Modular design ensures:**
1. ✓ Clean separation of concerns (binary, multi-class, hierarchical)
2. ✓ Code reuse (`qgfl_core.py` shared)
3. ✓ Easy extension (add new loss module, register, done)
4. ✓ Flexible experimentation (swap loss types via `--loss-type`)
5. ✓ Future-proof (staging, multi-species, DN-DETR ready)

**Current focus:** Phase 1 (Binary QGFL) - get this working first, then extend!


---

## APPENDIX: Transfer Learning Exploration (Future Work)

### PRIMARY APPROACH: Retrain from Scratch (Current Strategy)

All phases will retrain from ImageNet pretrained weights for clean, independent experiments.

### OPTIONAL: Transfer Learning Ablation (End of PhD)

After primary experiments complete, explore transfer learning as optional ablation study.

**Potential experiments:**
- Multi-species with binary transfer (best_binary_d3.pt → d3_multispecies)
- Staging with binary transfer (best_binary_d1.pt → d1_staging)
- Cross-dataset transfer studies

**Benefits:**
- Shows if binary knowledge helps multi-species/staging
- Faster training (100 vs 200 epochs)
- Interesting follow-up paper

**Current decision:** RETRAIN FROM SCRATCH (save transfer for later)
**Rationale:** Clean science, isolated QGFL contribution, easier to publish

**Action:** Save all best.pt files from each phase (enables future transfer experiments)

