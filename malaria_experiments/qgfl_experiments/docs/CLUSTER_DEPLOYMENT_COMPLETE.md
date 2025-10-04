# Cluster Deployment - Complete Summary

**Date**: October 4, 2025
**Status**: ✅ Successfully Deployed - Training Running
**Jobs**: 6 experiments (2 running, 4 queued)

---

## Deployment Overview

Successfully deployed all Phase 1 baseline experiments to UCD ADAPT cluster:

- **2 YOLO models**: YOLOv8s, YOLOv11s
- **3 datasets**: D1 (P. falciparum, 398), D2 (P. vivax, 1,328), D3 (Multi-species, 29,228)
- **Total**: 6 experiments running sequentially (2 concurrent max)

**Cluster Details:**
- Partition: `rtx8000` (default, 15-day time limit)
- GPUs: Quadro RTX 8000 (48GB VRAM)
- Nodes: ADAPT-CLIN, ML-01
- Environment: `phd_env` (Python 3.10)

---

## Critical Issues Fixed

### 1. **Hardcoded Path Issues**

**Problem**: Multiple files had hardcoded local paths that broke on cluster.

**Files Fixed**:
- ✅ `src/utils/paths.py` - Auto-detection logic for base path
- ✅ `src/utils/coco_to_yolo.py` - Added `base_path` parameter
- ✅ `cluster_run_baseline.py` - Fixed YAML path to use `Path(__file__).parent`

**Solution Pattern**:
```python
# BAD - Hardcoded
yaml_file = Path('../configs/data_yamls/d1_binary.yaml')

# GOOD - Portable
script_dir = Path(__file__).parent.absolute()
yaml_file = script_dir / 'configs' / 'data_yamls' / f'{config.dataset}_{config.task}.yaml'
```

### 2. **Symlink Path Resolution**

**Problem**: YOLO format used symlinks with absolute local paths (`/Users/thabangisaka/...`)

**Root Cause**:
- Symlinks created locally pointed to absolute paths
- When uploaded to cluster, paths didn't exist
- Training failed with "No such file or directory"

**Solution**: Recreated symlinks on cluster with **relative paths**:
```bash
# Fixed pattern for all datasets/splits
cd dataset_d{1,2,3}/yolo_format/{binary,species}/{train,val,test}/images
for link in *; do
  if [ -L "$link" ]; then
    target=$(basename "$link")
    rm "$link"
    ln -s "../../../../images/$target" "$link"
  fi
done
```

**Verified Counts**:
- D1: 225 train, 57 val, 116 test = 398 total ✅
- D2: 966 train, 242 val, 120 test = 1,328 total ✅
- D3: 20,830 train, 3,890 val, 4,508 test = 29,228 total ✅

### 3. **Visualization Symlink Resolution Bug**

**Problem**: `visualizer.py` line 233 used `.resolve()` on symlinks, converting to absolute paths.

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory:
'/Users/thabangisaka/.../dataset_d1/images/file.jpg'
```

**Solution**: Disabled visualization in `cluster_run_baseline.py`:
```python
# Line 187 - Skip visualization on cluster
print("Skipping dataset visualization on cluster to avoid symlink path issues...")
```

**Future Fix**: Modify `visualizer.py` to avoid `.resolve()` or handle symlinks better.

### 4. **Upload Exclusions**

**Problem**: Initial upload included `.venv` (2.6GB), `__pycache__`, etc.

**Solution**: Added rsync exclusions in `upload_to_cluster.sh`:
```bash
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='env/' \
  --exclude='phd_env/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='.git/' \
  --exclude='.DS_Store' \
  --exclude='runs/' \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa" \
  "/path/to/malaria_experiments/" \
  d23125116@147.252.6.50:~/malaria_qgfl_experiments/
```

**Size Savings**: ~2.6GB excluded

### 5. **SSH Legacy Key Exchange**

**Problem**: Cluster only supports old SSH algorithms:
```
Unable to negotiate with 147.252.6.20 port 22: no matching key exchange method found
```

**Solution**: Added legacy SSH options:
```bash
-o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa
```

### 6. **Cluster IP Correction**

**Problem**: Documentation had wrong IP (147.252.6.20)

**Correct IP**: `147.252.6.50`

### 7. **Partition Configuration**

**Problem**: Scripts used non-existent partition `MEDIUM-G2`

**Available Partitions**:
- `rtx8000` (default, 15-day limit) ✅ **SELECTED**
- `rtx40` (2-day limit, too short for D3)
- `ada` (5-day limit, nodes busy)

**Fix**: Updated all 6 SLURM scripts to use `--partition=rtx8000`

### 8. **Mixed Image Formats**

**Discovery**: Datasets use different formats:
- D1: `.jpg` files
- D2: `.png` (train/val) + `.jpg` (test)
- D3: `.jpg` files

**Verification**: YAML configs use generic paths (`train/images`), YOLO auto-detects both formats ✅

---

## Files Modified (Local)

### 1. `cluster_run_baseline.py`
**Changes**:
- Line 187: Disabled visualization (symlink issue)
- Line 366-367: Fixed YAML path to use `Path(__file__).parent`

### 2. `upload_to_cluster.sh`
**Changes**:
- Line 13: Fixed IP to 147.252.6.50
- Lines 11-20: Added rsync exclusions for venv, cache, etc.

### 3. SLURM Job Scripts (all 6)
**Changes**:
- Partition: `MEDIUM-G2` → `rtx8000`
- Python path: `cluster_run_baseline.py` → `qgfl_experiments/cluster_run_baseline.py`

---

## Cluster-Only Changes (Not in Git)

These changes were made directly on cluster and **should not** be committed:

1. **Symlink fixes** - Dataset-specific, cluster paths
2. **No netrc file** - W&B auto-created `/home/CAMPUS/d23125116/.netrc`

---

## Current Status

### Running Jobs

```
JOBID   PARTITION  NAME         USER      ST  TIME    NODES  NODELIST
161772  rtx8000    yolo_v8s_d1  d23125116 R   0:56    1      ADAPT-CLIN
161776  rtx8000    yolo_v8s_d3  d23125116 R   1:27    1      ADAPT-CLIN
```

### Queued Jobs

```
161773  rtx8000    yolo_v11s_d1  d23125116  PD  (AssocMaxJobsLimit)
161774  rtx8000    yolo_v8s_d2   d23125116  PD  (AssocMaxJobsLimit)
161775  rtx8000    yolo_v11s_d2  d23125116  PD  (AssocMaxJobsLimit)
161777  rtx8000    yolo_v11s_d3  d23125116  PD  (AssocMaxJobsLimit)
```

### Training Verification

✅ Model loaded: YOLOv8s (11.1M parameters)
✅ Optimizer: SGD(lr=0.005, momentum=0.95)
✅ AMP checks passed
✅ Data caches created
✅ Augmentations active: Blur, MedianBlur, ToGray, CLAHE
✅ W&B logging: https://wandb.ai/leraning/malaria_qgfl_experiments

### Log Files

```
logs/yolo_v8s_d1_161772.out    # D1 + YOLOv8s
logs/yolo_v8s_d1_161772.err
logs/yolo_v8s_d3_161776.out    # D3 + YOLOv8s
logs/yolo_v8s_d3_161776.err
```

---

## Key Learnings for Future Deployments

### 1. **Path Portability Checklist**
- ❌ Never use hardcoded absolute paths
- ✅ Always use `Path(__file__).parent` for script-relative paths
- ✅ Use environment detection (check CWD, search upward)
- ✅ Pass `base_path` as parameter to utility functions

### 2. **Symlink Best Practices**
- ✅ Always use **relative symlinks** for portability
- ✅ Test symlink resolution before upload
- ❌ Avoid `.resolve()` on symlinks in visualization code
- ✅ Verify symlinks work: `readlink <file>` should show `../../` not `/Users/`

### 3. **Upload Strategy**
- ✅ Always exclude: `.venv/`, `__pycache__/`, `.git/`, `runs/`
- ✅ Use rsync with `--dry-run` first to verify
- ✅ Test with small dataset before full upload
- ✅ Verify file counts after upload

### 4. **Cluster-Specific Adaptations**
- ✅ Check available partitions: `scontrol show partition`
- ✅ Test SSH connection before bulk upload
- ✅ Use legacy SSH options if cluster is old
- ✅ Disable visualization/plotting on headless clusters
- ✅ Set matplotlib backend to `Agg` before imports

### 5. **Dataset Verification**
- ✅ Always verify symlink counts match actual images
- ✅ Check both `.jpg` and `.png` files
- ✅ Test with small subset before full training
- ✅ Verify YAML paths are generic (no file extensions)

### 6. **Monitoring Best Practices**
```bash
# Queue status
watch -n 60 'squeue -u $USER'

# Training progress
cat logs/*.out | grep -i epoch | tail -n 20

# Job history
sacct -u $USER --format=JobID,JobName,State,Elapsed -n | tail -20

# W&B dashboard
echo "https://wandb.ai/leraning/malaria_qgfl_experiments"
```

---

## Timeline Estimates

| Dataset | Images | Est. Time per Epoch | Total (200 epochs) |
|---------|--------|---------------------|-------------------|
| D1      | 398    | 1-2 min            | 4-6 hours        |
| D2      | 1,328  | 2-4 min            | 8-12 hours       |
| D3      | 29,228 | 12-18 min          | 48-72 hours      |

**Total for all 6 jobs**: ~5-7 days (running 2 concurrent)

---

## Next Steps

1. ✅ Monitor training progress via W&B
2. ✅ Check logs periodically for errors
3. ⏳ Wait for all 6 experiments to complete
4. ⏳ Download results and analyze metrics
5. ⏳ Proceed to Phase 2: QGFL implementation

---

## Files Ready for Git Commit

**Modified Files** (ready to commit):
- `qgfl_experiments/cluster_run_baseline.py` - Path fixes + visualization skip
- `qgfl_experiments/upload_to_cluster.sh` - IP fix + exclusions
- `qgfl_experiments/docs/CLUSTER_DEPLOYMENT_COMPLETE.md` - This file

**Note**: Job scripts in `cluster_scripts/` were created on cluster and can be regenerated from `CLUSTER_DEPLOYMENT_STEPS.md`

---

## W&B Integration Verified

✅ API key configured in `configs/baseline_config.py` (line 47)
✅ Project: `malaria_qgfl_experiments`
✅ Auto-login successful on cluster
✅ Real-time logging active
✅ Model artifacts will be uploaded at end of training (Cell 19 logic preserved)

**Dashboard**: https://wandb.ai/leraning/malaria_qgfl_experiments

---

**Deployment completed successfully!** 🎉
