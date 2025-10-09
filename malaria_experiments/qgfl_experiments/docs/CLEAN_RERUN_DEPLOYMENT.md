# Clean Rerun Deployment Guide

## Overview
This guide covers the clean rerun of all baseline experiments (6 YOLO + 3 RT-DETR) with updated per-class mAP@50-95 extraction.

## Files Modified (Need Upload)
- `cluster_run_baseline.py` - Added per-class mAP@50-95 to W&B tables
- `src/evaluation/evaluator.py` - Extract per-class mAP from YOLO metrics
- `cluster_scripts/run_d1_rtdetr.sh` - NEW
- `cluster_scripts/run_d2_rtdetr.sh` - NEW
- `cluster_scripts/run_d3_rtdetr.sh` - NEW

## Step 1: Upload Updated Files to Cluster

```bash
# From local machine
bash upload_to_cluster.sh
```

This uploads entire `malaria_experiments/` folder to cluster at `~/malaria_qgfl_experiments/`

## Step 2: SSH into Cluster

```bash
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.50
```

## Step 3: Clean Old Results (IMPORTANT)

```bash
cd ~/malaria_qgfl_experiments

# Backup old results (optional)
# tar -czf old_results_$(date +%Y%m%d).tar.gz results/ wandb/ runs/

# Clean directories for fresh start
rm -rf results/
rm -rf wandb/
rm -rf runs/
rm -rf qgfl_experiments/*.log

# Create fresh logs directory
mkdir -p qgfl_experiments/cluster_scripts/logs

echo "✓ Cluster cleaned for fresh rerun"
```

## Step 4: Verify Uploaded Files

```bash
# Check critical files exist
ls -lh qgfl_experiments/cluster_run_baseline.py
ls -lh qgfl_experiments/src/evaluation/evaluator.py
ls -lh qgfl_experiments/cluster_scripts/run_d*.sh

# Verify all 9 SLURM scripts present
ls qgfl_experiments/cluster_scripts/run_*.sh | grep -v test
# Should show:
# run_d1_rtdetr.sh
# run_d1_yolov11s.sh
# run_d1_yolov8s.sh
# run_d2_rtdetr.sh
# run_d2_yolov11s.sh
# run_d2_yolov8s.sh
# run_d3_rtdetr.sh
# run_d3_yolov11s.sh
# run_d3_yolov8s.sh
```

## Step 5: Submit Experiments to Queue

**IMPORTANT**: Cluster allows only 2 simultaneous jobs. Submit in batches.

### Batch 1: D1 Experiments (2 jobs)
```bash
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts

sbatch run_d1_yolov8s.sh
sbatch run_d1_yolov11s.sh

# Check queue
squeue -u d23125116
```

### Batch 2: D1 RT-DETR + D2 Start (2 jobs)
Wait for at least one D1 YOLO to complete, then:
```bash
sbatch run_d1_rtdetr.sh
sbatch run_d2_yolov8s.sh
```

### Batch 3: D2 Continuation (2 jobs)
```bash
sbatch run_d2_yolov11s.sh
sbatch run_d2_rtdetr.sh
```

### Batch 4: D3 Experiments (2 jobs)
```bash
sbatch run_d3_yolov8s.sh
sbatch run_d3_yolov11s.sh
```

### Batch 5: D3 RT-DETR (1 job)
```bash
sbatch run_d3_rtdetr.sh
```

## Step 6: Monitor Progress

### Check Queue Status
```bash
squeue -u d23125116
```

### Watch Live Logs
```bash
# From logs directory
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs

# Watch latest log
tail -f $(ls -t *.out | head -1)
```

### Check W&B Dashboard
https://wandb.ai/learning/malaria_qgfl_experiments

All runs should appear with consistent per-class mAP@50-95 columns.

## Experiment Matrix

| Dataset | YOLOv8s | YOLOv11s | RT-DETR-L |
|---------|---------|----------|-----------|
| D1 (P. falciparum) | ✓ | ✓ | ✓ |
| D2 (P. vivax) | ✓ | ✓ | ✓ |
| D3 (Mixed species) | ✓ | ✓ | ✓ |

**Total**: 9 experiments

## Parameters (Consistent Across All Models)
- Epochs: 200
- Batch size: 16
- Learning rate: Default (model-specific)
- Weight decay: Default (model-specific)
- Augmentation: Default YOLO/RT-DETR settings

**Rationale**: Parameters validated from "sins of omission" paper parameter search. Using same hyperparameters for RT-DETR ensures fair comparison (architecture is the only variable).

## Expected Runtime
- YOLOv8s/YOLOv11s: ~16-20 hours per experiment
- RT-DETR-L: ~20-24 hours per experiment
- Total (sequential): ~7-9 days
- Total (2-job parallel): ~3-4 days

## Verification Checklist

After all experiments complete:

- [ ] All 9 runs appear in W&B dashboard
- [ ] Each run has `validation_per_class` table with mAP50-95 column
- [ ] Each run has `test_per_class` table with mAP50-95 column
- [ ] Each run has `precision_recall_analysis` table with mAP50-95 column
- [ ] Global mAP50-95 = mean(per-class mAP50-95) for each run
- [ ] All runs show 200 epochs completed
- [ ] Results directories contain evaluation JSON files
- [ ] No failed jobs in SLURM logs

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/[jobname]_[jobid].err

# Common issues:
# - Python environment not activated
# - Missing dependencies
# - Dataset paths incorrect
```

### Out of Memory
```bash
# Check memory usage in logs
grep -i "memory\|oom" logs/*.err

# If OOM, reduce batch size in SLURM script (line 16)
# Change: --batch-size 16 → --batch-size 8
```

### W&B Not Logging
```bash
# Check wandb login status on cluster
wandb status

# Re-login if needed
wandb login
```

### Dataset Not Found
```bash
# Verify dataset paths exist
ls ~/malaria_qgfl_experiments/dataset_d1/yolo_format/binary/
ls ~/malaria_qgfl_experiments/dataset_d2/yolo_format/binary/
ls ~/malaria_qgfl_experiments/dataset_d3/yolo_format/binary/
```

## Post-Completion

After all experiments finish:

1. Download results for local analysis
```bash
# From local machine
rsync -avz \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa" \
  d23125116@147.252.6.50:~/malaria_qgfl_experiments/results/ \
  ./malaria_experiments/cluster_results/
```

2. Generate comparison tables in notebook
3. Write paper results section
4. Create visualizations for publication

## Notes

- RT-DETR scripts use `rtdetr-l` model (large variant)
- All experiments use binary classification (Infected vs Uninfected)
- Datasets automatically detected in `~/malaria_qgfl_experiments/dataset_d{1,2,3}/`
- W&B project: `malaria_qgfl_experiments`
- W&B entity: `learning`
