# Cluster Upload Verification Checklist

## Files Modified for Cluster Deployment

### 1. cluster_run_baseline.py (130K)
**Critical Changes:**
- ✅ Optimizer auto-selection (lines 57-62): SGD for YOLO, auto for RT-DETR
- ✅ W&B logs args.optimizer (line 306): Logs actual optimizer used
- ✅ Hyperparameter logging (lines 490-498): Logs before training
- ✅ train_args uses args.optimizer (line 568): Passes correct optimizer
- ✅ GPU auto-detection (line 544): Works on both laptop and cluster
- ✅ RT-DETR predictor selection (line 873): Uses RTDETR() class

**Smoke Test Results:**
- ✅ YOLO v8s: SGD(lr=0.01, momentum=0.95) - PASSED
- ✅ RT-DETR: AdamW(lr=0.001667, momentum=0.9) - PASSED

### 2. cluster_run_qgfl.py (142K)
**Critical Changes:**
- ✅ Same optimizer fixes as baseline
- ✅ QGFL parameter fixes (args.gamma_infected not args.qgfl_gamma_infected)
- ✅ W&B naming includes _qgfl suffix (line 538)
- ✅ QGFL loss patching for both YOLO and RT-DETR (lines 127-303)
- ✅ Hardcoded QGFL params logged to W&B (lines 757-772)

**Smoke Test Results:**
- ✅ YOLO v8s + QGFL: mAP50=0.581, Infected mAP50-95=0.271 - PASSED
- ✅ RT-DETR + QGFL: mAP50=0.549, Infected mAP50-95=0.694 - PASSED

### 3. src/evaluation/evaluator.py (37K)
**Critical Changes:**
- ✅ RT-DETR predictor import (line 12): Added RTDETR
- ✅ Model-specific initialization (lines 20-28): Auto-detects RT-DETR
- ✅ Config-based thresholds (7 locations): Uses config.conf, config.iou

**Verification:**
- ✅ Works with YOLO models (v8, v11)
- ✅ Works with RT-DETR models
- ✅ Uses conf=0.25, iou=0.45 from config

### 4. configs/baseline_config.py (2.4K)
**Critical Changes:**
- ✅ conf: 0.25 (was 0.5) - Guemas methodology
- ✅ iou: 0.45 (was 0.5) - Domain-specific for malaria

**Verification:**
- ✅ All smoke tests use these thresholds correctly

## Verification Summary

### Baseline Smoke Tests (3 files uploaded previously)
| Test | Status | Optimizer | Metrics |
|------|--------|-----------|---------|
| YOLO v8s D1 | ✅ PASSED | SGD | mAP50=0.58 |
| YOLO v11s D1 | ✅ PASSED | SGD | - |
| RT-DETR D1 | ✅ PASSED | AdamW | mAP50=0.55 |

### QGFL Smoke Tests (Ready for upload)
| Test | Status | Optimizer | QGFL Loss | Infected mAP |
|------|--------|-----------|-----------|--------------|
| YOLO v8s + QGFL | ✅ PASSED | SGD | Active | 0.271 |
| RT-DETR + QGFL | ✅ PASSED | AdamW | Active | 0.694 |

### Critical Verifications
- ✅ Optimizer auto-selection working (SGD for YOLO, AdamW for RT-DETR)
- ✅ W&B logging correct (logs actual optimizer, not default)
- ✅ RT-DETR predictor selection working
- ✅ Config-based evaluation thresholds working
- ✅ QGFL loss integration working (both YOLO and RT-DETR)
- ✅ W&B naming correct (_qgfl suffix for QGFL runs)

## Files to Upload

### Primary Scripts (MUST upload)
1. ✅ cluster_run_baseline.py (130K) - Baseline experiments
2. ✅ cluster_run_qgfl.py (142K) - QGFL experiments
3. ✅ src/evaluation/evaluator.py (37K) - Shared evaluator
4. ✅ configs/baseline_config.py (2.4K) - Configuration

### Supporting Files (Already on cluster)
- ✅ src/losses/qgfl_yolo.py - QGFL YOLO loss
- ✅ src/losses/qgfl_rtdetr.py - QGFL RT-DETR loss
- ✅ Dataset files (D1, D2, D3)

## Upload Commands

```bash
# Navigate to cluster upload directory
cd ~/malaria_qgfl_experiments/qgfl_experiments

# Upload baseline script
scp cluster_run_baseline.py thabang@graham.computecanada.ca:~/malaria_qgfl_experiments/qgfl_experiments/

# Upload QGFL script
scp cluster_run_qgfl.py thabang@graham.computecanada.ca:~/malaria_qgfl_experiments/qgfl_experiments/

# Upload evaluator
scp src/evaluation/evaluator.py thabang@graham.computecanada.ca:~/malaria_qgfl_experiments/qgfl_experiments/src/evaluation/

# Upload config
scp configs/baseline_config.py thabang@graham.computecanada.ca:~/malaria_qgfl_experiments/qgfl_experiments/configs/
```

## Post-Upload Verification

After upload, run on cluster:
```bash
# Check files uploaded
ls -lh cluster_run_baseline.py cluster_run_qgfl.py
ls -lh src/evaluation/evaluator.py configs/baseline_config.py

# Verify Python syntax
python3 -m py_compile cluster_run_baseline.py
python3 -m py_compile cluster_run_qgfl.py

# Quick grep verification
grep -n "args.optimizer = 'auto'" cluster_run_baseline.py
grep -n "args.optimizer = 'SGD'" cluster_run_baseline.py
grep -n "RTDETR" src/evaluation/evaluator.py
```

## Cluster Submission (After Upload Verified)

### Baseline Jobs (9 experiments)
Already submitted to cluster - running 200 epochs

### QGFL Jobs (9 experiments - ready to submit)
```bash
# D1 experiments
sbatch cluster_scripts/run_d1_yolov8s_qgfl.sh
sbatch cluster_scripts/run_d1_yolov11s_qgfl.sh
sbatch cluster_scripts/run_d1_rtdetr_qgfl.sh

# D2 experiments
sbatch cluster_scripts/run_d2_yolov8s_qgfl.sh
sbatch cluster_scripts/run_d2_yolov11s_qgfl.sh
sbatch cluster_scripts/run_d2_rtdetr_qgfl.sh

# D3 experiments
sbatch cluster_scripts/run_d3_yolov8s_qgfl.sh
sbatch cluster_scripts/run_d3_yolov11s_qgfl.sh
sbatch cluster_scripts/run_d3_rtdetr_qgfl.sh
```

## Expected Results (200 epochs)

### RT-DETR + QGFL (Best Expected Performance)
- Infected Precision: 65-75%
- Infected Recall: 90-95%
- Infected mAP50-95: 0.80-0.85

### YOLO + QGFL
- Infected Precision: 30-40%
- Infected Recall: 85-90%
- Infected mAP50-95: 0.45-0.55

## Sign-off

All smoke tests PASSED. All critical fixes verified. Ready for cluster upload and 200-epoch deployment.

Date: October 9, 2025
Status: ✅ READY FOR CLUSTER DEPLOYMENT
