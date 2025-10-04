# Cluster Setup Guide - QGFL Malaria Experiments

**Date:** 2025-10-03
**Based on:** Previous successful cluster.txt template

---

## Cluster Environment Details (From Template)

**Cluster Type:** SLURM
**Partition:** MEDIUM-G2
**User:** d23125116
**Home:** `/home/CAMPUS/d23125116/`
**Virtual Env:** `~/phd_env/bin/activate`
**GPU:** 1 GPU per job
**Memory:** 32GB
**CPUs:** 4 per task

---

## Step 1: Environment Setup on Cluster

### 1.1 Create/Verify Virtual Environment

```bash
# If phd_env doesn't exist or needs update
cd ~
python3 -m venv phd_env
source ~/phd_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install wandb
pip install pandas numpy matplotlib seaborn
pip install scikit-learn
pip install opencv-python
pip install pillow
pip install tqdm
pip install tabulate
pip install pyyaml

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import wandb; print('W&B OK')"
```

### 1.2 Setup W&B API Key

```bash
# Set W&B API key as environment variable
echo "export WANDB_API_KEY='your_key_here'" >> ~/.bashrc
source ~/.bashrc

# Or login interactively
wandb login
```

---

## Step 2: Upload Project to Cluster

### 2.1 Directory Structure on Cluster

```
/home/CAMPUS/d23125116/malaria_qgfl_experiments/
├── cluster_run_baseline.py          # Converted from notebook
├── configs/
│   ├── baseline_config.py
│   └── data_yamls/
│       ├── d1_binary.yaml
│       ├── d2_binary.yaml
│       └── d3_binary.yaml
├── src/
│   ├── evaluation/
│   │   └── evaluator.py
│   ├── training/
│   │   └── strategy_wrapper.py
│   └── utils/
│       ├── paths.py
│       ├── coco_to_yolo.py
│       └── visualizer.py
├── datasets/                        # Or symlink to shared storage
│   ├── dataset_d1/
│   ├── dataset_d2/
│   └── dataset_d3/
├── models/
│   ├── yolov8s.pt
│   └── yolov11s.pt
└── cluster_scripts/
    ├── run_d1_yolov8s.sh
    ├── run_d1_yolov11s.sh
    ├── run_d2_yolov8s.sh
    ├── run_d2_yolov11s.sh
    ├── run_d3_yolov8s.sh
    └── run_d3_yolov11s.sh
```

### 2.2 Upload Commands (from local machine)

```bash
# Option 1: rsync (recommended)
rsync -avz --progress \
  ~/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/qgfl_experiments/ \
  d23125116@cluster-address:/home/CAMPUS/d23125116/malaria_qgfl_experiments/

# Option 2: scp
scp -r ~/Downloads/thabang_phd/Experiments/Year\ 3\ Experiments/malaria_experiments/qgfl_experiments/ \
  d23125116@cluster-address:/home/CAMPUS/d23125116/malaria_qgfl_experiments/
```

---

## Step 3: Convert Notebook to Cluster Script

### 3.1 Conversion Command (on cluster or local, then upload)

```bash
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/notebooks/

# Convert notebook to Python script
jupyter nbconvert --to python \
  --PythonExporter.exclude_markdown=True \
  --PythonExporter.exclude_output=True \
  01_run_baseline.ipynb \
  --output ../cluster_run_baseline.py

# Verify conversion
ls -lh ../cluster_run_baseline.py
head -50 ../cluster_run_baseline.py
```

### 3.2 Add Argparse to Converted Script

After conversion, manually add argparse at the top (or I'll create pre-converted version):

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QGFL Malaria Detection - Cluster Training Script
Converted from: 01_run_baseline.ipynb
"""

import argparse
import os
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLO Baseline Training for Malaria Detection')
parser.add_argument('--dataset', type=str, required=True, choices=['d1', 'd2', 'd3'],
                    help='Dataset to use (d1, d2, or d3)')
parser.add_argument('--model', type=str, required=True, choices=['yolov8s', 'yolov11s'],
                    help='Model architecture')
parser.add_argument('--task', type=str, default='binary', choices=['binary', 'species', 'staging'],
                    help='Classification task')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size')
parser.add_argument('--use-wandb', action='store_true', default=True,
                    help='Use Weights & Biases logging')
parser.add_argument('--results-dir', type=str, default='../results',
                    help='Base directory for results')
args = parser.parse_args()

# Rest of notebook code with config using args...
```

---

## Step 4: Create SLURM Job Scripts

### Template for Each Job

```bash
#!/bin/bash
#SBATCH --job-name=yolo_{MODEL}_{DATASET}
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_{MODEL}_{DATASET}_%j.out
#SBATCH --error=logs/yolo_{MODEL}_{DATASET}_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=48:00:00

# Activate virtual environment
source ~/phd_env/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export WANDB_API_KEY=$WANDB_API_KEY

# Print environment info
echo "========================================="
echo "Job: YOLO {MODEL} on {DATASET}"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="
echo "Python version:"
python --version
echo "PyTorch CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo "GPU Info:"
nvidia-smi
echo "========================================="

# Change to working directory
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

# Create logs directory if needed
mkdir -p logs results

# Run training
echo "Starting training: {MODEL} on {DATASET}..."
python -u cluster_run_baseline.py \
    --dataset {DATASET} \
    --model {MODEL} \
    --task binary \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb

echo "========================================="
echo "Training completed!"
echo "Results saved to: results/"
echo "========================================="
```

### 4.1 Individual Job Scripts

**File: `cluster_scripts/run_d1_yolov8s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v8s_d1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v8s_d1_%j.out
#SBATCH --error=logs/yolo_v8s_d1_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv8s on D1 (P. falciparum, 398 images)"
python -u cluster_run_baseline.py --dataset d1 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
```

**File: `cluster_scripts/run_d1_yolov11s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v11s_d1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v11s_d1_%j.out
#SBATCH --error=logs/yolo_v11s_d1_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv11s on D1 (P. falciparum, 398 images)"
python -u cluster_run_baseline.py --dataset d1 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
```

**File: `cluster_scripts/run_d2_yolov8s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v8s_d2
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v8s_d2_%j.out
#SBATCH --error=logs/yolo_v8s_d2_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=32:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv8s on D2 (P. vivax, 1,328 images)"
python -u cluster_run_baseline.py --dataset d2 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
```

**File: `cluster_scripts/run_d2_yolov11s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v11s_d2
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v11s_d2_%j.out
#SBATCH --error=logs/yolo_v11s_d2_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=32:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv11s on D2 (P. vivax, 1,328 images)"
python -u cluster_run_baseline.py --dataset d2 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
```

**File: `cluster_scripts/run_d3_yolov8s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v8s_d3
#SBATCH --gres=gpu:1
#SBATCH --mem=64000  # Increased for D3 (large dataset)
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/yolo_v8s_d3_%j.out
#SBATCH --error=logs/yolo_v8s_d3_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=72:00:00  # 3 days for large dataset

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv8s on D3 (Multi-species, 28,905 images)"
python -u cluster_run_baseline.py --dataset d3 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
```

**File: `cluster_scripts/run_d3_yolov11s.sh`**
```bash
#!/bin/bash
#SBATCH --job-name=yolo_v11s_d3
#SBATCH --gres=gpu:1
#SBATCH --mem=64000  # Increased for D3 (large dataset)
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/yolo_v11s_d3_%j.out
#SBATCH --error=logs/yolo_v11s_d3_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=72:00:00  # 3 days for large dataset

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/

echo "Job: YOLOv11s on D3 (Multi-species, 28,905 images)"
python -u cluster_run_baseline.py --dataset d3 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
```

---

## Step 5: Master Submission Script

**File: `cluster_scripts/submit_all_jobs.sh`**
```bash
#!/bin/bash
# Submit all 6 baseline experiments

echo "========================================="
echo "Submitting QGFL Baseline Experiments"
echo "========================================="

cd /home/CAMPUS/d23125116/malaria_qgfl_experiments/cluster_scripts/

# Make all scripts executable
chmod +x *.sh

# Create logs directory
mkdir -p ../logs

# Submit jobs
echo "Submitting D1 jobs..."
JOB1=$(sbatch run_d1_yolov8s.sh | awk '{print $4}')
echo "  - YOLOv8s on D1: Job ID $JOB1"

JOB2=$(sbatch run_d1_yolov11s.sh | awk '{print $4}')
echo "  - YOLOv11s on D1: Job ID $JOB2"

echo "Submitting D2 jobs..."
JOB3=$(sbatch run_d2_yolov8s.sh | awk '{print $4}')
echo "  - YOLOv8s on D2: Job ID $JOB3"

JOB4=$(sbatch run_d2_yolov11s.sh | awk '{print $4}')
echo "  - YOLOv11s on D2: Job ID $JOB4"

echo "Submitting D3 jobs..."
JOB5=$(sbatch run_d3_yolov8s.sh | awk '{print $4}')
echo "  - YOLOv8s on D3: Job ID $JOB5"

JOB6=$(sbatch run_d3_yolov11s.sh | awk '{print $4}')
echo "  - YOLOv11s on D3: Job ID $JOB6"

echo "========================================="
echo "All jobs submitted!"
echo "Job IDs: $JOB1 $JOB2 $JOB3 $JOB4 $JOB5 $JOB6"
echo "========================================="

# Show queue status
echo "Current queue status:"
squeue -u d23125116

echo ""
echo "Monitor jobs with:"
echo "  squeue -u d23125116"
echo "  tail -f logs/yolo_*.out"
```

---

## Step 6: Monitoring Commands

### Check Job Status
```bash
# View all your jobs
squeue -u d23125116

# View specific job
squeue -j JOB_ID

# View detailed job info
scontrol show job JOB_ID
```

### Monitor Output
```bash
# Follow output in real-time
tail -f logs/yolo_v8s_d1_JOBID.out

# Check for errors
tail -f logs/yolo_v8s_d1_JOBID.err

# View all recent output
ls -lt logs/
```

### Cancel Jobs
```bash
# Cancel specific job
scancel JOB_ID

# Cancel all your jobs
scancel -u d23125116

# Cancel by name pattern
scancel --name=yolo_v8s_d1
```

---

## Step 7: Results Collection

### After Jobs Complete

```bash
# Check results directory
ls -lh /home/CAMPUS/d23125116/malaria_qgfl_experiments/results/

# Download results to local machine
rsync -avz --progress \
  d23125116@cluster:/home/CAMPUS/d23125116/malaria_qgfl_experiments/results/ \
  ~/Downloads/thabang_phd/Experiments/cluster_results/

# Or specific experiment
rsync -avz --progress \
  d23125116@cluster:/home/CAMPUS/d23125116/malaria_qgfl_experiments/results/d1_binary_yolov8s_noweights_20250103/ \
  ~/Downloads/thabang_phd/Experiments/cluster_results/
```

---

## Step 8: Troubleshooting

### Common Issues

**1. GPU Not Available**
```bash
# Check GPU allocation
nvidia-smi

# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"
```

**2. Out of Memory**
```bash
# Reduce batch size in job script
--batch-size 8  # Instead of 16
```

**3. Module Not Found**
```bash
# Reinstall in virtual env
source ~/phd_env/bin/activate
pip install missing_module
```

**4. File Not Found**
```bash
# Check paths are absolute
ls -la /home/CAMPUS/d23125116/malaria_qgfl_experiments/
```

**5. W&B Login Issues**
```bash
# Set API key
export WANDB_API_KEY='your_key'
wandb login
```

---

## Expected Timeline

| Experiment | Dataset Size | Estimated Time | Resources |
|------------|--------------|----------------|-----------|
| D1 YOLOv8s | 398 images | ~4-6 hours | 32GB, 1 GPU |
| D1 YOLOv11s | 398 images | ~4-6 hours | 32GB, 1 GPU |
| D2 YOLOv8s | 1,328 images | ~8-12 hours | 32GB, 1 GPU |
| D2 YOLOv11s | 1,328 images | ~8-12 hours | 32GB, 1 GPU |
| D3 YOLOv8s | 28,905 images | ~48-72 hours | 64GB, 1 GPU |
| D3 YOLOv11s | 28,905 images | ~48-72 hours | 64GB, 1 GPU |

**Total:** ~6 jobs running concurrently (if resources available) or sequentially

---

## Checklist Before Submission

- [ ] Virtual environment created and tested
- [ ] All dependencies installed
- [ ] W&B API key configured
- [ ] Project uploaded to cluster
- [ ] Notebook converted to cluster_run_baseline.py
- [ ] Argparse added to script
- [ ] All 6 job scripts created and executable
- [ ] Dataset paths verified on cluster
- [ ] Model weights (.pt files) uploaded
- [ ] Logs directory created
- [ ] Test run with 2 epochs successful
- [ ] W&B logging verified working

---

**Status:** Ready to execute once cluster access confirmed
