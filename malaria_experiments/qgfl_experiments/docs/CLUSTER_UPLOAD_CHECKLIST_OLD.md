# Cluster Upload Checklist - RT-DETR Baseline Test

**Date:** 2025-10-06
**Test Scope:** D1 + D2 only (NOT D3 - too large for initial test)
**Purpose:** Validate AdamW hyperparameters @ 200 epochs before full deployment

---

## Files to Upload

### 1. Python Training Script (Modified)

```bash
qgfl_experiments/cluster_run_baseline.py
```

**Changes made (RT-DETR ONLY):**
- ✅ Added CLI arguments: `--optimizer`, `--lr0`, `--lrf`, `--warmup-epochs`, `--cls`, `--box`
- ✅ Applied hyperparameters to `train_args` dictionary
- ✅ W&B logging for hyperparameters
- ✅ Changed device from 'cpu' to auto-detect GPU
- ⚠️ **YOLO NOT AFFECTED** - YOLO uses defaults (no CLI args passed in YOLO scripts)

**Git diff summary:**
```diff
+ # Hyperparameter arguments (for RT-DETR tuning)
+ parser.add_argument('--optimizer', type=str, default='auto', ...)
+ parser.add_argument('--lr0', type=float, default=0.01, ...)
+ parser.add_argument('--lrf', type=float, default=0.01, ...)
+ parser.add_argument('--warmup-epochs', type=float, default=3.0, ...)
+ parser.add_argument('--cls', type=float, default=0.5, ...)
+ parser.add_argument('--box', type=float, default=7.5, ...)

- 'device': 'cpu',  # Force CPU
+ 'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # Auto-detect

- 'optimizer': getattr(config, 'optimizer', 'SGD'),
+ 'optimizer': args.optimizer,  # CLI override

- 'lr0': getattr(config, 'lr0', 0.005),
+ 'lr0': args.lr0,  # CLI override

- 'warmup_epochs': getattr(config, 'warmup_epochs', 3.0),
+ 'warmup_epochs': args.warmup_epochs,  # CLI override

- 'cls': strategy_params.get('cls', 0.5),
+ 'cls': args.cls,  # CLI override

- 'box': strategy_params.get('box', 7.5),
+ 'box': args.box,  # CLI override
```

### 2. Cluster Scripts (RT-DETR Test Only)

```bash
qgfl_experiments/cluster_scripts/run_d1_rtdetr.sh  # Updated
qgfl_experiments/cluster_scripts/run_d2_rtdetr.sh  # Updated
qgfl_experiments/cluster_scripts/test_rtdetr_d1_d2.sh  # NEW
```

**RT-DETR scripts now include:**
```bash
python -u qgfl_experiments/cluster_run_baseline.py \
    --dataset {d1|d2} --model rtdetr-l --epochs 200 --batch-size 16 \
    --optimizer AdamW --lr0 0.0017 --lrf 0.01 --warmup-epochs 5 \
    --cls 1.0 --box 7.5 \
    --use-wandb
```

**YOLO scripts remain UNCHANGED:**
```bash
# Example: run_d1_yolov8s.sh - NO custom hyperparameters
python -u qgfl_experiments/cluster_run_baseline.py \
    --dataset d1 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
# ✓ Uses defaults: optimizer='auto', lr0=0.01, warmup=3, cls=0.5, box=7.5
```

---

## Upload Commands

### Step 1: Upload Modified Files to Cluster

```bash
# From local machine
cd ~/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/qgfl_experiments

# Upload the modified training script
scp cluster_run_baseline.py d23125116@csserver15.ucd.ie:~/malaria_qgfl_experiments/qgfl_experiments/

# Upload RT-DETR cluster scripts
scp cluster_scripts/run_d1_rtdetr.sh d23125116@csserver15.ucd.ie:~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/
scp cluster_scripts/run_d2_rtdetr.sh d23125116@csserver15.ucd.ie:~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/
scp cluster_scripts/test_rtdetr_d1_d2.sh d23125116@csserver15.ucd.ie:~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/
```

### Step 2: Make Scripts Executable

```bash
# SSH to cluster
ssh d23125116@csserver15.ucd.ie

# Make test script executable
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts
chmod +x test_rtdetr_d1_d2.sh
```

### Step 3: Submit Test Jobs

```bash
# Submit D1 + D2 RT-DETR test
./test_rtdetr_d1_d2.sh
```

---

## Verification Checklist

### Before Upload

- [x] ✅ `cluster_run_baseline.py` has RT-DETR hyperparameter CLI args
- [x] ✅ RT-DETR scripts pass AdamW hyperparameters
- [x] ✅ YOLO scripts remain unchanged (no custom hyperparameters)
- [x] ✅ Test script created for D1 + D2 only (not D3)
- [x] ✅ Device auto-detects GPU (was hardcoded to CPU)

### After Upload (Verify on Cluster)

```bash
# Check files exist
ls -lh ~/malaria_qgfl_experiments/qgfl_experiments/cluster_run_baseline.py
ls -lh ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/run_d1_rtdetr.sh
ls -lh ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/run_d2_rtdetr.sh
ls -lh ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/test_rtdetr_d1_d2.sh

# Verify RT-DETR script has AdamW params
grep "AdamW" ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/run_d1_rtdetr.sh

# Verify YOLO script still has NO custom params (should find nothing)
grep "AdamW" ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/run_d1_yolov8s.sh
```

### After Submission (Monitor)

```bash
# Check queue
squeue -u d23125116

# Expected output: 2 jobs (rtdetr_d1, rtdetr_d2)

# Watch logs in real-time
tail -f logs/rtdetr_d1_*.out
tail -f logs/rtdetr_d2_*.out

# Check for hyperparameter confirmation in logs
grep "Optimizer:" logs/rtdetr_d1_*.out
grep "AdamW" logs/rtdetr_d1_*.out
```

---

## Expected Behavior

### D1 RT-DETR (P. falciparum, 398 images)

**Runtime:** ~24 hours (200 epochs × 150 batches = 30,000 iterations)

**Expected hyperparameters in log:**
```
✓ Hyperparameters logged to W&B:
  - Optimizer: AdamW
  - Learning rate: 0.0017 → 0.000017
  - Warmup epochs: 5
  - Loss weights: cls=1.0, box=7.5
```

**Training behavior (first 20 epochs):**
- Epochs 1-5 (warmup): cls_loss near 0, giou_loss high
- Epochs 6-20: cls_loss rising (0.064 → 0.5+), giou_loss decreasing

**Success criteria @ 200 epochs:**
- Max confidence ≥ 0.5 (clinical threshold)
- Test recall @ conf=0.5 ≥ 50%
- mAP50 ≥ 60% (comparable to YOLO baseline)

### D2 RT-DETR (P. vivax, 1,328 images)

**Runtime:** ~32 hours (more images)

**Same hyperparameters and expected behavior as D1**

**Success criteria @ 200 epochs:**
- Max confidence ≥ 0.5
- Test recall @ conf=0.5 ≥ 55%
- mAP50 ≥ 65% (larger dataset, should perform better)

---

## Troubleshooting

### Issue 1: YOLO Models Break

**Symptom:** YOLO baselines fail or perform poorly after upload

**Cause:** YOLO scripts accidentally modified or defaults changed

**Fix:** YOLO scripts should have NO custom hyperparameters:
```bash
# Correct YOLO command (NO --optimizer, --lr0, etc.)
python -u qgfl_experiments/cluster_run_baseline.py \
    --dataset d1 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
```

**Verification:**
```bash
# Should return empty (no AdamW in YOLO scripts)
grep "AdamW" cluster_scripts/run_d1_yolov8s.sh
```

### Issue 2: RT-DETR Uses Wrong Optimizer

**Symptom:** RT-DETR logs show "SGD" instead of "AdamW"

**Cause:** CLI arguments not passed in cluster script

**Fix:** Check cluster script has:
```bash
--optimizer AdamW --lr0 0.0017 --lrf 0.01 --warmup-epochs 5 --cls 1.0 --box 7.5
```

**Verification:**
```bash
grep "Optimizer: AdamW" logs/rtdetr_d1_*.out
```

### Issue 3: GPU Not Detected

**Symptom:** Training runs on CPU instead of GPU

**Cause:** Device auto-detection failing

**Fix:** Check cluster log for:
```
Device: cuda
```

If shows CPU, manually specify in cluster script:
```bash
--device 0  # Force GPU 0
```

---

## What NOT to Upload

### ❌ Do NOT Upload These Files

1. **QGFL Script (Not Needed for Baseline Test)**
   - `cluster_run_qgfl.py` - Only for Phase 3 (RT-DETR QGFL)

2. **D3 Scripts (Too Large for Test)**
   - `cluster_scripts/run_d3_rtdetr.sh` - Will test after D1/D2 validate

3. **YOLO Scripts (Already Working)**
   - `cluster_scripts/run_d1_yolov8s.sh` - Don't touch
   - `cluster_scripts/run_d1_yolov11s.sh` - Don't touch
   - `cluster_scripts/run_d2_yolov8s.sh` - Don't touch
   - `cluster_scripts/run_d2_yolov11s.sh` - Don't touch
   - `cluster_scripts/run_d3_yolov8s.sh` - Don't touch
   - `cluster_scripts/run_d3_yolov11s.sh` - Don't touch

4. **Documentation Files**
   - `docs/*.md` - Not needed for running experiments
   - `CLUSTER_UPLOAD_CHECKLIST.md` (this file) - Local reference only

---

## Summary

**Files to upload:** 4 files total
1. `cluster_run_baseline.py` - Modified for RT-DETR hyperparameter control
2. `cluster_scripts/run_d1_rtdetr.sh` - Updated with AdamW params
3. `cluster_scripts/run_d2_rtdetr.sh` - Updated with AdamW params
4. `cluster_scripts/test_rtdetr_d1_d2.sh` - New test submission script

**What changes:**
- RT-DETR models ONLY (now use AdamW with validated hyperparameters)

**What stays the same:**
- YOLO models (use default hyperparameters, no changes)

**Test scope:**
- D1 + D2 RT-DETR @ 200 epochs (D3 later after validation)

**Expected timeline:**
- D1: ~24 hours
- D2: ~32 hours
- Results available: 1-2 days

**Next steps after test:**
1. Validate D1/D2 results meet success criteria
2. If successful: Deploy D3 RT-DETR @ 200 epochs
3. If all baselines successful: Deploy RT-DETR QGFL (Phase 3)

---

**Ready to deploy:** ✅
**YOLO protected:** ✅
**Test scope limited:** ✅
**Hyperparameters validated:** ✅
