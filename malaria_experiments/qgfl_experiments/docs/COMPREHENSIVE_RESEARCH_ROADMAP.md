# Comprehensive Research Roadmap: From Baseline to Publication

**Date**: October 4, 2025
**Project**: Quality-Guided Focal Loss for Malaria Detection
**Status**: Phase 1 Running (Baseline Experiments)

---

## Understanding the QGFL Paper Methodology

### Paper's Progressive Framework (5 Levels)

The published QGFL paper tested a **progressive adaptation approach**:

1. **Level 1**: Standard Focal Loss (FL) - α=0.9, γ=2.0
2. **Level 2**: Difficulty-Aware (DA) - Dynamic γ adjustment
3. **Level 3**: Class-Difficulty (CD) - Class-specific γ (infected=8.0, uninfected=4.0)
4. **Level 4**: Class-Difficulty + Threshold (CD+T) - Difficulty threshold=0.925
5. **Level 5**: Complete QGFL - Quality-guided + UIoU integration

**Key Finding from Paper**: The paper tested ALL progressive levels to show systematic improvement, not just baseline vs final QGFL.

### Critical Research Question

**Do we implement all stages or jump to final QGFL?**

**Answer**: Based on the paper's methodology and research rigor, we should:
- **Implement all progressive stages** for:
  - Reproducibility of published results
  - Ablation study showing contribution of each component
  - Understanding which components work best for your specific datasets
  - Academic rigor (reviewers will expect this)

- **Final comparison** should be: Baseline vs FL vs DA vs CD vs CD+T vs QGFL

---

## Current Phase 1: Baseline Experiments (NOW RUNNING)

### What's Running (October 2025)
```
✅ 6 Experiments Submitted (200 epochs each):
1. YOLOv8s + D1 (binary)
2. YOLOv11s + D1 (binary)
3. YOLOv8s + D2 (binary)
4. YOLOv11s + D2 (binary)
5. YOLOv8s + D3 (binary)
6. YOLOv11s + D3 (binary)

Architecture: RetinaNet (as per paper)
Loss: Default Focal Loss (library defaults)
Task: Binary detection (infected vs uninfected)
Datasets: 3 (D1, D2, D3)
Models: 2 (YOLOv8s, YOLOv11s)
```

### Expected Outputs (5-7 days)
- Model weights (best.pt, last.pt)
- Training metrics (mAP50, mAP50-95, precision, recall, F1)
- Per-class metrics (infected vs uninfected performance)
- Prevalence-stratified analysis (0-1%, 1-3%, 3-5%, >5%)
- TIDE error analysis
- W&B logged artifacts

---

## Phase 2: QGFL Progressive Implementation (AFTER BASELINE COMPLETES)

### Phase 2A: Standard Focal Loss Tuning (~1 week)

**Objective**: Establish FL baseline with tuned hyperparameters (Level 1)

**Experiments** (6 total):
```python
# Configuration
Loss: Standard Focal Loss
α: 0.9 (infected), 0.1 (uninfected)  # Paper values
γ: 2.0                                # Paper value

Models: YOLOv8s, YOLOv11s
Datasets: D1, D2, D3
Task: Binary
Epochs: 200
```

**Expected Improvement**: 5-10% mAP over baseline (per paper: D1=+6.9%, D2=+8.2%, D3=+4.3%)

### Phase 2B: Difficulty-Aware Scaling (~1 week)

**Objective**: Test dynamic γ adjustment (Level 2)

**Experiments** (6 total):
```python
# Configuration
Loss: Difficulty-Aware Focal Loss
γ_eff = γ + (max_γ - γ) × difficulty
max_γ: 4.0
difficulty: (1 - p_t)

Models: YOLOv8s, YOLOv11s
Datasets: D1, D2, D3
```

**Expected**: May show mixed results (paper showed regression on some datasets)

### Phase 2C: Class-Difficulty Scaling (~1 week)

**Objective**: Class-specific focusing parameters (Level 3)

**Experiments** (6 total):
```python
# Configuration
Loss: Class-Difficulty Focal Loss
infected_max_γ: 8.0    # Higher focus on infected
uninfected_max_γ: 4.0  # Lower focus on uninfected

Models: YOLOv8s, YOLOv11s
Datasets: D1, D2, D3
```

**Expected**: Better than DA, targets minority class specifically

### Phase 2D: Threshold Integration (~1 week)

**Objective**: Add difficulty thresholding (Level 4)

**Experiments** (6 total):
```python
# Configuration
Loss: Class-Difficulty + Threshold
difficulty_threshold: 0.925
difficulty_adjusted = max(raw_difficulty - threshold, 0) / (1 - threshold)

Models: YOLOv8s, YOLOv11s
Datasets: D1, D2, D3
```

**Expected**: Best performance before full QGFL (paper: D1=+6.9%, D2=+8.6%)

### Phase 2E: Complete QGFL (~1-2 weeks)

**Objective**: Full quality-guided framework (Level 5)

**Experiments** (6 total):
```python
# Configuration
Loss: Complete QGFL
- Quality-guided weighting
- UIoU integration
- All previous components

quality_margin: 0.5
quality_factor: 2.0
uiou_ratio: 2.0 → 0.5 (linear decay)

Models: YOLOv8s, YOLOv11s
Datasets: D1, D2, D3
```

**Expected**: Best overall (paper: D1 1-3% density +46% recall, D2 +93%, D3 +8%)

**Total Phase 2 Duration**: ~5-6 weeks
**Total Experiments**: 30 (5 loss variants × 2 models × 3 datasets)

---

## Phase 3: Transformer Baselines (PARALLEL TRACK)

### Why Transformers?

1. **Paper's Future Work**: "future validation will involve implementing QGFL across state-of-the-art architectures, including **YOLO variants and transformer-based models**"

2. **Current SOTA for Medical Imaging**:
   - RT-DETR (Real-Time DETR) - Already used in malaria detection [Guemas et al., 2024]
   - DINO/DINOv2 - Self-supervised vision transformers
   - Hybrid approaches (DINOv2 backbone + DETR head)

### Phase 3A: RT-DETR Baseline (~2 weeks)

**Objective**: Establish transformer baseline

**Experiments** (6 total):
```python
# Configuration
Architecture: RT-DETR (Real-Time Detection Transformer)
Backbone: ResNet-50 or PVT-v2
Loss: Standard DETR loss (default)

Models: RT-DETR-R50, RT-DETR-R101
Datasets: D1, D2, D3
Task: Binary
Epochs: 200
```

**Why RT-DETR?**
- Proven for malaria detection (Guemas et al., 2024)
- Real-time capable
- Directly comparable to YOLO

### Phase 3B: DINOv2 Feature Extraction (~2 weeks)

**Objective**: Test foundation model features

**Experiments** (3 total):
```python
# Configuration
Architecture: DINOv2 (ViT-B/14 or ViT-L/14) + Detection Head
Backbone: DINOv2 pretrained (frozen or fine-tuned)
Head: Simple detection head (FCOS/RetinaNet style)

Models: DINOv2-B, DINOv2-L
Datasets: D1, D2, D3 (combined for more data)
Task: Binary
Epochs: 100 (frozen backbone), 50 (fine-tuned)
```

**Why DINOv2?**
- Self-supervised pretraining on diverse images
- Strong feature representation
- Could handle morphological similarity better

### Phase 3C: Hybrid Transformer-DETR (~2 weeks)

**Objective**: Best transformer architecture

**Experiments** (3 total):
```python
# Configuration
Architecture: DINOv2 backbone + DETR/RT-DETR head
Backbone: DINOv2-ViT-B/14
Neck: Feature Pyramid or hybrid CNN-Transformer
Head: DETR or RT-DETR decoder

Datasets: D1, D2, D3
Task: Binary
Epochs: 200
```

**Total Phase 3 Duration**: ~6-8 weeks
**Total Experiments**: 12

---

## Phase 4: QGFL on Transformers (~4-6 weeks)

### Objective: Test QGFL generalizability across architectures

**Critical Research Question**: Does QGFL work on transformer-based detectors?

**Approach**: Apply **only the best QGFL variant** (Level 4 or Level 5) to transformers

**Experiments** (12 total):
```python
# Configuration
Architectures: RT-DETR, DINOv2+DETR
Loss Variants:
  - Standard (baseline)
  - QGFL (Level 5)
  - OR CD+Thresh (Level 4) if Level 5 doesn't improve transformers

Datasets: D1, D2, D3
Models: 2 transformer architectures
Epochs: 200
```

**Expected Outcome**:
- If QGFL improves transformers → Strong generalizability claim
- If not → Architecture-specific analysis needed

---

## Phase 5: Hierarchical Tasks (Species & Staging) (~6-8 weeks)

### Background

**Current**: Binary detection (infected vs uninfected)
**Clinical Need**:
- **Species identification**: P. falciparum, P. vivax, P. malariae, P. ovale
- **Stage identification**: Ring, Trophozoite, Schizont, Gametocyte

### Phase 5A: Species Detection (Multi-Class)

**Dataset Availability**:
- D1: Only P. falciparum (single species) ❌
- D2: Only P. vivax (single species) ❌
- D3: Multi-species (P. falciparum, P. vivax, mixed) ✅

**Experiments** (8 total):
```python
# Configuration
Task: Species Detection (4-5 classes)
Classes:
  - Uninfected
  - P. falciparum
  - P. vivax
  - P. malariae (if available)
  - P. ovale (if available)

Architecture: Best from Phase 2 + Best Transformer from Phase 3
Loss: Baseline → QGFL
Dataset: D3 only (only multi-species dataset)
Models: YOLOv8s, YOLOv11s, RT-DETR, DINOv2-DETR
Epochs: 200
```

**Challenge**: D3 species annotations need verification

### Phase 5B: Stage Detection (Hierarchical Multi-Class)

**Complexity**: Life cycle stages are morphologically similar

**Experiments** (8 total):
```python
# Configuration
Task: Stage Detection (hierarchical)
Classes:
  Level 1: Infected vs Uninfected (binary)
  Level 2 (if infected): Ring, Trophozoite, Schizont, Gametocyte

Architecture: Hierarchical approach
  - Option 1: Two-stage detector (binary → stage)
  - Option 2: Multi-task learning (joint binary + stage)
  - Option 3: Hierarchical loss (weighted binary + stage)

Loss: QGFL adapted for hierarchical classes
Dataset: D1 + D2 + D3 (if stage annotations available)
Models: Best 2 from previous phases
Epochs: 200
```

**Data Requirement**: Check if datasets have stage annotations

---

## Recommended Implementation Order

### Timeline: 6-8 Months Total

```
Month 1 (NOW): Phase 1 - Baseline Running
    Week 1-2: Training (200 epochs)
    Week 3: Analysis, W&B review
    Week 4: Paper writing (baseline results)

Month 2-3: Phase 2 - QGFL Progressive Implementation
    Week 5-6: FL + DA + CD (3 variants × 6 experiments = 18 runs)
    Week 7-8: CD+T + QGFL (2 variants × 6 experiments = 12 runs)
    Week 9-10: Analysis, comparison, ablation study
    Week 11-12: Paper writing (QGFL results)

Month 4-5: Phase 3 - Transformer Baselines
    Week 13-14: RT-DETR baseline
    Week 15-16: DINOv2 experiments
    Week 17-18: Hybrid transformer
    Week 19-20: Analysis + paper writing

Month 6: Phase 4 - QGFL on Transformers
    Week 21-22: QGFL + RT-DETR
    Week 23-24: QGFL + DINOv2-DETR
    Week 25: Analysis + paper writing

Month 7-8: Phase 5 - Hierarchical Tasks
    Week 26-27: Species detection (D3)
    Week 28-29: Stage detection (if annotations available)
    Week 30-32: Final analysis, paper writing, thesis integration
```

### Parallel Execution Strategy

**To accelerate**, run experiments in parallel:

```
Month 2-3: Phase 2 (QGFL) + Phase 3 (Transformers) in PARALLEL
  - QGFL on YOLO models (use current cluster)
  - Transformer baselines (can run on different GPU or local)

Month 4: Phase 4 only (requires Phase 2 + 3 complete)

Month 5-6: Phase 5 (if dataset annotations ready)
```

---

## What to Include in Publications

### Paper 1: Binary Detection with QGFL (Primary Contribution)

**Title**: "Quality-Guided Focal Loss: Progressive Adaptation for Minority Class Detection in Malaria Microscopy"

**Content**:
- ✅ Baseline (RetinaNet)
- ✅ FL (Standard Focal Loss)
- ✅ All 5 progressive QGFL levels
- ✅ Ablation study
- ✅ 3 datasets (D1, D2, D3)
- ✅ 2 YOLO models (YOLOv8s, YOLOv11s)
- ✅ Prevalence-stratified analysis (1-3% density - KEY CLINICAL METRIC)

**Status**: In progress (Month 1-3)

### Paper 2: Transformer Architectures + QGFL Generalizability

**Title**: "Cross-Architecture Validation of Quality-Guided Focal Loss: From CNNs to Transformers for Medical Object Detection"

**Content**:
- RT-DETR baseline
- DINOv2 features
- Hybrid architectures
- QGFL on transformers
- Generalizability analysis

**Status**: Future (Month 4-6)

### Paper 3: Hierarchical Multi-Class Detection

**Title**: "Hierarchical Quality-Guided Detection: Species and Stage Identification in Plasmodium Microscopy"

**Content**:
- Species detection (4-5 classes)
- Stage detection (hierarchical)
- Multi-task learning
- Clinical workflow integration

**Status**: Future (Month 7-8)

---

## Critical Decisions Needed NOW

### Decision 1: All Progressive Stages or Jump to QGFL?

**Recommendation**: **Implement all 5 progressive stages**

**Rationale**:
- Paper used this methodology
- Shows systematic improvement
- Ablation study for publication
- Identifies best components for your datasets
- Academic rigor

### Decision 2: Which Architectures for Phase 2 (QGFL)?

**Options**:
A. YOLO only (YOLOv8s, YOLOv11s) - **Current baseline**
B. RetinaNet (as per paper) + YOLO
C. YOLO + RT-DETR

**Recommendation**: **Option A (YOLO only) for Phase 2**

**Rationale**:
- Baselines already running on YOLO
- Faster training (real-time models)
- Defer transformers to Phase 3
- Can validate on RetinaNet later if needed

### Decision 3: Foundation Models (DINOv2) Priority?

**Recommendation**: **Medium priority (Phase 3B)**

**Rationale**:
- Interesting research direction
- Strong feature representation
- But not essential for core QGFL contribution
- Can be explored after QGFL proven on YOLO

### Decision 4: Species/Staging Tasks - Now or Later?

**Recommendation**: **Later (Phase 5) AFTER binary QGFL validated**

**Rationale**:
- Need to verify D3 has species/stage annotations
- Binary detection is foundation
- Hierarchical tasks build on binary success
- Can be separate paper

---

## Immediate Next Steps (This Week)

### 1. Monitor Baseline Training
- Check W&B dashboard daily
- Verify no job failures
- Track mAP50 progression

### 2. Prepare QGFL Implementation
- Review cluster_run_baseline.py
- Identify where to add QGFL loss functions
- Create modular loss implementation:
  ```python
  # losses/focal_loss.py
  class StandardFocalLoss()
  class DifficultyAwareFocalLoss()
  class ClassDifficultyFocalLoss()
  class ThresholdFocalLoss()
  class QGFL()  # Complete
  ```

### 3. Plan Phase 2 Experiments
- Create experiment config files for all 5 QGFL levels
- Update SLURM job scripts
- Prepare W&B project structure (separate runs for each loss variant)

### 4. Dataset Annotation Verification
- **Action**: Check D3 dataset for species/stage labels
- **Command**:
  ```python
  # On cluster or local
  import json
  from pathlib import Path

  # Check COCO annotations
  d3_annotations = Path("dataset_d3/annotations/instances_train.json")
  with open(d3_annotations) as f:
      data = json.load(f)

  # Check categories
  print("Categories:", data['categories'])
  # If only 2 classes → binary only
  # If 4-5 classes → species available
  # If >5 classes → species + staging possibly available
  ```

---

## Resource Planning

### Computational Requirements

**Current Cluster**:
- 2 concurrent jobs max (rtx8000 partition)
- Quadro RTX 8000 GPUs (48GB VRAM)
- Enough for YOLO models

**Phase 2 (QGFL - 30 experiments)**:
- Duration: ~5-6 weeks (2 concurrent)
- Can accelerate by using multiple partitions if available

**Phase 3-4 (Transformers - 24 experiments)**:
- Transformers need more VRAM
- May need larger GPUs or reduced batch size
- RT-DETR: similar to YOLO (should fit)
- DINOv2: larger (may need batch size reduction)

**Phase 5 (Hierarchical - 16 experiments)**:
- Multi-class detection = more VRAM
- Likely need batch size tuning

### Storage Requirements

**Per Experiment**:
- Model weights: ~50-100MB
- Results/logs: ~10-50MB
- Total per experiment: ~150MB

**Total Storage Estimate**:
- Phase 1: 6 experiments × 150MB = ~1GB
- Phase 2: 30 experiments × 150MB = ~5GB
- Phase 3-4: 24 experiments × 150MB = ~4GB
- Phase 5: 16 experiments × 150MB = ~2.5GB
- **Grand Total**: ~15-20GB (manageable)

---

## Final Recommendations

### Core Research Path (Highest Priority)

```
1. ✅ Phase 1: Baseline (CURRENT - RUNNING)
2. 🔥 Phase 2A-E: QGFL Progressive (NEXT - CRITICAL)
3. 🔥 Phase 3A: RT-DETR Baseline (PARALLEL if resources allow)
4. ⚡ Phase 4: QGFL on Transformers (GENERALIZABILITY)
5. 📊 Phase 5: Species Detection (CLINICAL IMPACT)
```

### Extended Research Path (If Time/Resources Allow)

```
6. DINOv2 Foundation Model Experiments
7. Stage Detection (Hierarchical Multi-Class)
8. Hybrid Architectures (CNN+Transformer)
9. Cross-Dataset Generalization Studies
```

### Publication Strategy

**Primary Paper** (Target: Top Medical AI Conference/Journal):
- Binary detection with complete QGFL framework
- All 5 progressive levels + ablation
- 3 datasets, prevalence-stratified analysis
- Target: MICCAI 2026, TMI, Medical Image Analysis

**Secondary Paper** (Architecture Generalization):
- QGFL on transformers
- Cross-architecture validation
- Target: CVPR 2026 Medical AI Workshop, Pattern Recognition

**Thesis Chapter Structure**:
- Chapter 1: Introduction + Background
- Chapter 2: Baseline Experiments + Evaluation Framework
- Chapter 3: QGFL Progressive Implementation (CORE)
- Chapter 4: Transformer Architectures
- Chapter 5: Hierarchical Tasks (Species/Staging)
- Chapter 6: Conclusions + Future Work

---

## Questions for You

1. **QGFL Stages**: Agree to implement all 5 progressive stages (FL, DA, CD, CD+T, QGFL)?

2. **Transformer Priority**: Should we run RT-DETR baseline in parallel with Phase 2, or wait until QGFL complete?

3. **Dataset Annotations**: Can you verify if D3 has species/stage labels, or just binary?

4. **Publication Target**: Aiming for MICCAI 2026 (June deadline) or TMI (journal, no deadline)?

5. **Cluster Access**: Is the rtx8000 partition the only one available, or can we use others for parallel execution?

---

**This roadmap prioritizes scientific rigor, reproducibility, and clinical impact while being realistic about computational resources and timelines.**
