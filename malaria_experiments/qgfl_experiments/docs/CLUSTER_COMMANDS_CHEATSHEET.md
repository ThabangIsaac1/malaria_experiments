# Cluster Commands Cheatsheet

Quick reference for monitoring and managing baseline experiments on the cluster.

---

## 1. SSH INTO CLUSTER

```bash
ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 d23125116@147.252.6.50
```

---

## 2. SUBMIT ALL 9 EXPERIMENTS

```bash
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts

sbatch run_d1_yolov8s.sh
sbatch run_d1_yolov11s.sh
sbatch run_d1_rtdetr.sh
sbatch run_d2_yolov8s.sh
sbatch run_d2_yolov11s.sh
sbatch run_d2_rtdetr.sh
sbatch run_d3_yolov8s.sh
sbatch run_d3_yolov11s.sh
sbatch run_d3_rtdetr.sh
```

--- QGFL
sbatch run_d1_rtdetr_qgfl.sh
sbatch run_d1_yolov11s_qgfl.sh
sbatch run_d1_yolov8s_qgfl.sh
sbatch run_d2_rtdetr_qgfl.sh
sbatch run_d2_yolov11s_qgfl.sh
sbatch run_d2_yolov8s_qgfl.sh
sbatch run_d3_rtdetr_qgfl.sh
sbatch run_d3_yolov11s_qgfl.sh
sbatch run_d3_yolov8s_qgfl.sh




## 3. CHECK JOB QUEUE

### Show all your jobs
```bash
squeue -u d23125116
```

**Output columns:**
- `JOBID` - Job identifier (e.g., 12345)
- `PARTITION` - Queue (rtx8000)
- `NAME` - Job name (yolo_v8s_d1, rtdetr_d2, etc.)
- `USER` - Your username
- `ST` - Status (PD=pending, R=running, CG=completing)
- `TIME` - Runtime so far
- `NODES` - Number of nodes
- `NODELIST(REASON)` - Which node OR why pending

### Show detailed job info
```bash
scontrol show job JOBID
```

### Show only running jobs
```bash
squeue -u d23125116 -t RUNNING
```

### Show only pending jobs
```bash
squeue -u d23125116 -t PENDING
```

---

## 4. MONITOR LOGS

### Go to logs directory
```bash
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs
```

### List all logs (sorted by time, newest first)
```bash
ls -lht
```

### Watch latest output log in real-time
```bash
tail -f $(ls -t *_*.out | head -1)
```

### Watch specific job output
```bash
tail -f yolo_v8s_d1_12345.out
```

### Watch specific job errors
```bash
tail -f yolo_v8s_d1_12345.err
```

### Check last 50 lines of log
```bash
tail -50 yolo_v8s_d1_12345.out
```

### Search logs for errors
```bash
grep -i "error\|fail\|exception" *.out
grep -i "error\|fail\|exception" *.err
```

### Check for OOM (out of memory) errors
```bash
grep -i "memory\|oom\|cuda out of memory" *.err
```

### Find W&B run URLs in logs
```bash
grep "wandb.ai" *.out
```

---

## 5. CANCEL JOBS

### Cancel specific job
```bash
scancel JOBID
```

### Cancel all your jobs
```bash
scancel -u d23125116
```

### Cancel specific job by name pattern
```bash
scancel -n yolo_v8s_d1
```

---

## 6. CHECK DISK SPACE

### Check home directory usage
```bash
du -sh ~/malaria_qgfl_experiments
```

### Check detailed breakdown
```bash
du -h --max-depth=1 ~/malaria_qgfl_experiments | sort -hr
```

### Check available disk space
```bash
df -h ~
```

---

## 7. MONITOR GPU USAGE (while job is running)

### SSH into the compute node where job is running
First get node name from `squeue -u d23125116`, then:
```bash
ssh NODE_NAME  # e.g., ssh adapt-gpu01
```

### Check GPU status
```bash
nvidia-smi
```

### Watch GPU usage in real-time (updates every 2 seconds)
```bash
watch -n 2 nvidia-smi
```

### Exit GPU monitoring
Press `Ctrl+C` to stop watch, then `exit` to leave node

---

## 8. CHECK EXPERIMENT RESULTS

### List completed experiments
```bash
ls -lh ~/malaria_qgfl_experiments/results/
```

### Check specific experiment output
```bash
ls -lh ~/malaria_qgfl_experiments/results/yolov8s_d1_binary/
```

### View evaluation JSON
```bash
cat ~/malaria_qgfl_experiments/results/yolov8s_d1_binary/evaluation/results.json | python3 -m json.tool | less
```

### Check W&B runs directory
```bash
ls -lh ~/malaria_qgfl_experiments/wandb/
```

---

## 9. CHECK WANDB STATUS

### Verify W&B is logged in
```bash
wandb status
```

### Re-login if needed
```bash
wandb login
```

### List W&B runs
```bash
wandb runs learning/malaria_qgfl_experiments
```

---

## 10. TROUBLESHOOTING

### Job pending too long?
```bash
squeue -u d23125116 -t PENDING
# Check NODELIST(REASON) column for why
```

**Common reasons:**
- `Resources` - Waiting for GPU/CPU/memory
- `Priority` - Other jobs have higher priority
- `Dependency` - Waiting for another job to finish

### Job failed immediately?
```bash
# Check error log
cat logs/JOB_NAME_JOBID.err

# Check SLURM output
cat logs/JOB_NAME_JOBID.out
```

### Python environment issues?
```bash
# Verify environment exists
ls ~/phd_env/bin/activate

# Test environment manually
source ~/phd_env/bin/activate
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import wandb; print(wandb.__version__)"
```

### Dataset not found?
```bash
# Verify datasets exist
ls ~/malaria_qgfl_experiments/dataset_d1/yolo_format/binary/
ls ~/malaria_qgfl_experiments/dataset_d2/yolo_format/binary/
ls ~/malaria_qgfl_experiments/dataset_d3/yolo_format/binary/
```

### Check recent job history
```bash
sacct -u d23125116 --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
```

### Check only today's jobs
```bash
sacct -u d23125116 -S $(date +%Y-%m-%d) --format=JobID,JobName,State,ExitCode,Elapsed
```

---

## 11. DOWNLOAD RESULTS TO LOCAL MACHINE

### From your local terminal (not cluster)
```bash
# Download all results
rsync -avz \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa" \
  d23125116@147.252.6.50:~/malaria_qgfl_experiments/results/ \
  ~/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/cluster_results/

# Download logs only
rsync -avz \
  -e "ssh -o KexAlgorithms=+diffie-hellman-group14-sha1 -o HostKeyAlgorithms=+ssh-rsa" \
  d23125116@147.252.6.50:~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs/ \
  ~/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/cluster_logs/
```

---

## 12. CLEAN UP AFTER EXPERIMENTS

### Remove old runs (CAREFUL!)
```bash
rm -rf ~/malaria_qgfl_experiments/results/
rm -rf ~/malaria_qgfl_experiments/wandb/
rm -rf ~/malaria_qgfl_experiments/runs/
rm -f ~/malaria_qgfl_experiments/qgfl_experiments/*.log
```

### Archive instead of delete
```bash
cd ~/malaria_qgfl_experiments
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz results/ wandb/ runs/
rm -rf results/ wandb/ runs/
```

---

## 13. QUICK STATUS CHECK (RUN THIS OFTEN)

```bash
# One-liner to check everything
echo "=== JOB QUEUE ===" && squeue -u d23125116 && \
echo "" && echo "=== LATEST LOGS ===" && \
ls -lht ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs/ | head -5 && \
echo "" && echo "=== DISK USAGE ===" && \
du -sh ~/malaria_qgfl_experiments
```

---

## 14. EXPERIMENT TRACKING

### Expected runtime per experiment
- **YOLOv8s**: ~16-20 hours
- **YOLOv11s**: ~16-20 hours
- **RT-DETR-L**: ~20-24 hours

### Total experiments: 9
- 3 datasets (D1, D2, D3)
- 3 models per dataset (YOLOv8s, YOLOv11s, RT-DETR-L)

### W&B Dashboard
https://wandb.ai/learning/malaria_qgfl_experiments

### Expected W&B tables per run
1. `validation_per_class` - Per-class validation metrics (with mAP50-95)
2. `test_per_class` - Per-class test metrics (with mAP50-95, AP)
3. `precision_recall_analysis` - PR curve analysis (with mAP50-95, AP)

---

## 15. EMERGENCY COMMANDS

### System is slow/hanging?
```bash
# Check system load
uptime

# Check processes
top
# Press 'q' to quit

# Check your processes specifically
ps aux | grep d23125116
```

### Kill stuck Python process
```bash
# Find process ID
ps aux | grep "cluster_run_baseline.py"

# Kill it
kill -9 PID
```

### Check cluster status
```bash
sinfo
```

---

## COMMON WORKFLOWS

### Workflow 1: Submit and Monitor
```bash
# 1. Submit jobs
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts
sbatch run_d1_yolov8s.sh

# 2. Check submission
squeue -u d23125116

# 3. Watch log in real-time
cd logs
tail -f $(ls -t *.out | head -1)
```

### Workflow 2: Check All Running Jobs
```bash
# Show queue
squeue -u d23125116

# Show latest logs for each
cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs
for log in $(ls -t *.out | head -3); do
  echo "=== $log (last 20 lines) ==="
  tail -20 $log
  echo ""
done
```

### Workflow 3: Verify Experiment Completed Successfully
```bash
# 1. Check job finished
sacct -j JOBID

# 2. Check error log is empty
cat logs/JOB_NAME_JOBID.err

# 3. Verify results created
ls -lh ~/malaria_qgfl_experiments/results/EXPERIMENT_NAME/

# 4. Check W&B logged
grep "wandb.ai" logs/JOB_NAME_JOBID.out
```

---

## NOTES

- **Cluster limit**: Maximum 2 simultaneous jobs
- **Partition**: `rtx8000` (GPU partition)
- **GPU per job**: 1
- **Memory per job**: 32GB
- **CPUs per job**: 4
- **Max time**: 48 hours per job
- **Parameters**: epochs=200, batch_size=16 (all experiments)

---

## HELPFUL ALIASES (Add to ~/.bashrc on cluster)

```bash
# Add these to ~/.bashrc for quick access
alias qq='squeue -u d23125116'
alias logs='cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs'
alias watchlog='tail -f $(ls -t ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts/logs/*.out | head -1)'
alias exps='cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts'
alias results='cd ~/malaria_qgfl_experiments/results && ls -lh'
```

After adding, reload with: `source ~/.bashrc`

Then use:
- `qq` - Quick queue check
- `logs` - Jump to logs directory
- `watchlog` - Watch latest log
- `exps` - Jump to experiment scripts
- `results` - View results directory
