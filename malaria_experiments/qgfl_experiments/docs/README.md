# Documentation Index

This folder contains all project documentation to save context and maintain clarity.

## Setup & Status Documents

1. **[PHASE1_SETUP_COMPLETE.md](PHASE1_SETUP_COMPLETE.md)**
   - Overall Phase 1 status
   - What's been accomplished
   - How to use the scripts
   - Next steps

2. **[NOTEBOOK_ANALYSIS.md](NOTEBOOK_ANALYSIS.md)**
   - Complete breakdown of 01_run_baseline.ipynb
   - Cell-by-cell structure
   - Dependencies and file interactions
   - Collision points identified

3. **[CRITICAL_FINDINGS.md](CRITICAL_FINDINGS.md)**
   - Audit results: train_baseline.py vs notebook
   - Missing evaluation components
   - Impact assessment
   - What needs to be added

4. **[SOLUTION_ARCHITECTURE.md](SOLUTION_ARCHITECTURE.md)**
   - Decided approach: Single script with full evaluation
   - W&B integration rationale
   - Priority order for additions
   - Implementation plan

## Progress Tracking

### Completed ✅
- Dataset conversion (D1, D2, D3)
- YAML configurations
- Code cleanup (removed WeightedBaseline)
- Basic training script
- Cluster submission scripts

### In Progress 🔄
- Adding evaluation components cell-by-cell:
  - [ ] Cell 14: Prevalence-stratified analysis (CRITICAL)
  - [ ] Cell 13: Per-class analysis
  - [ ] Cell 19: Comprehensive W&B logging
  - [ ] Cell 15: PR curves
  - [ ] Cell 16: TIDE errors
  - [ ] Other visualization cells

### Pending ⏳
- Smoke testing
- Cluster deployment
- Phase 2 (RT-DETR, QGFL)

## Quick Reference

**Key Files:**
- Main training: `../train_baseline.py`
- Cluster submit: `../submit_experiments.sh`
- Reference notebook: `../notebooks/01_run_baseline.ipynb`

**Dataset Locations:**
- D1: `../../dataset_d1/yolo_format/binary/` (398 images)
- D2: `../../dataset_d2/yolo_format/binary/` (1,328 images)
- D3: `../../dataset_d3/yolo_format/binary/` (28,905 images)

**W&B Project:** malaria_qgfl_experiments

---

*Last updated: 2025-10-03*
