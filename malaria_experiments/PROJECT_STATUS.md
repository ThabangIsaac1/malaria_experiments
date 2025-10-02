# Malaria QGFL Experiments - Project Status

**Last Updated:** October 3, 2025
**Researcher:** Thabang Isaka
**Repository:** https://github.com/ThabangIsaac1/malaria_experiments

---

## Research Overview

**Title:** Quality-Guided Focal Loss for Malaria Parasite Detection Across Modern Object Detection Architectures

**Duration:** 12 weeks (3 phases)
**Contribution:** First systematic QGFL evaluation across YOLO variants and transformers + multi-class medical imaging extension

---

## Current Status: Phase 1 Foundation ✅

### What's Working

✅ **Dataset Infrastructure**
- D1: 398 images (P. falciparum) - Binary, Species, Staging ready
- D2: 1,328 images (P. vivax) - Binary, Species, Staging ready
- D3: 28,905 images (Multi-species) - Binary, Species ready
- All datasets in YOLO format with centralized images

✅ **Code Framework**
- Complete evaluation pipeline with TIDE analysis
- Precision-Recall curves with full curve data
- Prevalence-stratified analysis (critical for clinical relevance)
- Training strategy wrapper (NoWeights, WeightedBaseline, QGFL)
- W&B integration with comprehensive logging

✅ **Validated Experiments**
- E1.1 (YOLOv8s, D1, Binary, NoWeights) - **COMPLETE**
  - Model trained and evaluated
  - Results logged to W&B
  - Evaluation metrics confirmed

### Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset loaders | ✅ Complete | All 3 datasets working |
| YOLO baseline (no weights) | ✅ Complete | Validated on D1 |
| Weighted baseline | 🔄 Code ready | Needs testing |
| QGFL implementation | 🔄 Code ready | Needs validation |
| Multi-class QGFL | ⏳ Planned | Phase 2 |
| RT-DETR integration | ⏳ Planned | Phase 1 Week 3 |
| RedDino enhancement | ⏳ Planned | Phase 3 |

---

## Phase 1: Binary Classification (Weeks 1-4)

### Experimental Matrix (18 experiments)

| Exp ID | Architecture | Dataset | Task | Loss Function | Priority | Status |
|--------|-------------|---------|------|---------------|----------|--------|
| E1.1 | YOLOv8s | D1 | Binary | Standard FL | High | ✅ Complete |
| E1.2 | YOLOv8s | D1 | Binary | QGFL | High | 🔜 Next |
| E1.3 | YOLOv8s | D2 | Binary | Standard FL | High | ⏳ |
| E1.4 | YOLOv8s | D2 | Binary | QGFL | High | ⏳ |
| E1.5 | YOLOv8s | D3 | Binary | Standard FL | High | ⏳ |
| E1.6 | YOLOv8s | D3 | Binary | QGFL | High | ⏳ |
| E1.7-E1.12 | YOLOv11s | D1,D2,D3 | Binary | FL/QGFL | Medium | ⏳ |
| E1.13-E1.18 | RT-DETR-R18 | D1,D2,D3 | Binary | FL/QGFL | Medium | ⏳ |

---

## Technical Stack

### Implemented
- **Python:** 3.11
- **PyTorch:** 2.0+
- **Ultralytics:** Latest (YOLOv8, YOLOv11)
- **Weights & Biases:** Experiment tracking
- **Git:** Version control with GitHub

### Evaluation Metrics
1. **Global:** mAP@0.5, mAP@[0.5:0.95], Precision, Recall
2. **Per-Class:** Precision, Recall, F1, Support, TP/FP/FN
3. **PR Analysis:** AP, Optimal threshold, Max F1, Full PR curves
4. **Prevalence-Stratified:** Performance at 0-1%, 1-3%, 3-5%, >5% infection density
5. **TIDE Errors:** Classification, Localization, Duplicate, Background, Missed
6. **Confusion Matrix:** Object-level analysis

---

## Project Structure

```
malaria_experiments/
├── qgfl_experiments/
│   ├── configs/
│   │   ├── baseline_config.py          ✅ Complete
│   │   └── data_yamls/
│   │       └── d1_binary.yaml          ✅ Complete
│   ├── src/
│   │   ├── evaluation/
│   │   │   └── evaluator.py            ✅ Complete
│   │   ├── training/
│   │   │   └── strategy_wrapper.py     ✅ Complete
│   │   └── utils/
│   │       ├── coco_to_yolo.py         ✅ Complete
│   │       ├── paths.py                ✅ Complete
│   │       └── visualizer.py           ✅ Complete
│   ├── notebooks/
│   │   └── 01_run_baseline.ipynb       ✅ Complete
│   ├── results/                        (Not tracked - logged to W&B)
│   └── weights/                        (Not tracked - logged to W&B)
├── dataset_d1/                         ✅ Ready
├── dataset_d2/                         ✅ Ready
├── dataset_d3/                         ✅ Ready
└── Corrected Dataset Setups - Experiments.docx  ✅ Complete
```

---

## Next Steps (In Priority Order)

### This Week
1. **Test Weighted Baseline** - Validate `WeightedBaselineStrategy`
2. **Implement QGFL** - Complete E1.2 (YOLOv8s + QGFL on D1)
3. **Run Phase 1 validation set** - E1.1-E1.6 with 10 epochs each
4. **Prepare cluster scripts** - SLURM templates for batch submission

### Week 2
5. **YOLOv11s experiments** - E1.7-E1.12
6. **Compare YOLO variants** - Analyze architecture differences

### Week 3
7. **RT-DETR implementation** - E1.13-E1.18
8. **Phase 1 analysis** - Complete binary classification results

---

## Known Issues & Decisions

### Resolved
- ✅ Per-class mAP50 logging (decided to use W&B only, not git-tracked tables)
- ✅ Results storage (W&B cloud, not git repository)
- ✅ Git repository setup with proper .gitignore

### Open Questions
- [ ] Optimal QGFL hyperparameters for each dataset?
- [ ] Multi-class QGFL class-difficulty scaling strategy?
- [ ] RT-DETR integration complexity assessment?

---

## Publication Targets

### Primary Contributions
1. First systematic QGFL evaluation across modern architectures
2. Novel multi-class QGFL adaptation for hierarchical tasks
3. Foundation model integration (RedDino + QGFL)
4. Clinical evaluation framework (prevalence-stratified analysis)

### Target Venues
- **Medical Imaging:** MICCAI, Medical Image Analysis
- **Computer Vision:** CVPR, ICCV, ECCV
- **AI in Healthcare:** Nature Digital Medicine, JAMIA

---

## Research Timeline

| Week | Focus | Deliverables | Status |
|------|-------|--------------|--------|
| Week 1 | YOLOv8s Binary Foundation | E1.1-E1.6 complete | 🔄 In Progress |
| Week 2 | YOLOv11s Binary Extension | E1.7-E1.12 complete | ⏳ Planned |
| Week 3 | RT-DETR Binary | E1.13-E1.18 complete | ⏳ Planned |
| Week 4 | Phase 1 Analysis | Binary results analysis | ⏳ Planned |
| Week 5-6 | Multi-Class Species | Phase 2 experiments | ⏳ Planned |
| Week 7-8 | Multi-Class Staging | Phase 2 complete | ⏳ Planned |
| Week 9-10 | RedDino Integration | Phase 3 experiments | ⏳ Planned |
| Week 11 | Advanced Analysis | Cross-dataset, stratified | ⏳ Planned |
| Week 12 | Documentation | Paper preparation | ⏳ Planned |

---

## Git Workflow

### Protected Files (Tracked)
- Source code (`src/`)
- Configurations (`configs/`)
- Notebooks (without outputs)
- Documentation (`.md`, `.docx`)

### Excluded (Not Tracked)
- Datasets (too large, managed separately)
- Results (logged to W&B)
- Model weights (logged to W&B)
- Training runs (`runs/`, `wandb/`)
- Virtual environments (`.venv/`)

### Recent Commits
- `a370103` - Phase 1 Foundation: YOLO Baseline Framework
- `bd16444` - Pre-processing files for datasets

---

## Contact & Collaboration

**Researcher:** Thabang Isaka
**Institution:** [Your University]
**GitHub:** https://github.com/ThabangIsaac1/malaria_experiments
**Weights & Biases:** malaria_qgfl_experiments project

---

**End of Status Document**
