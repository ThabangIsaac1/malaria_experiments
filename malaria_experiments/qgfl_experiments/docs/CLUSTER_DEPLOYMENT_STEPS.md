# Cluster Deployment - Step-by-Step Commands

**Date:** 2025-10-03
**Cluster:** 147.252.6.20
**User:** d23125116

⚠️ **SECURITY NOTE:** Change your password after deployment!

---

## Resource Constraint

**Cluster allows:** 2 jobs maximum running concurrently
**Our experiments:** 6 total jobs (2 models × 3 datasets)

**Strategy:** Submit all 6, they will queue and run 2 at a time automatically

---

## Step 1: Upload Project to Cluster

### From your local terminal (with VPN on):

```bash
# Create directory on cluster (you'll be prompted for password)
ssh d23125116@147.252.6.20 "mkdir -p ~/malaria_qgfl_experiments"

# Upload entire project (this will take a few minutes)
rsync -avz --progress \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/" \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/

# Verify upload
ssh d23125116@147.252.6.20 "ls -lh ~/malaria_qgfl_experiments/"
```

**What gets uploaded:**
- All datasets (D1, D2, D3 in YOLO format)
- cluster_run_baseline.py
- configs/ (including W&B key)
- src/
- Model weights (yolov8s.pt, yolov11s.pt)

---

## Step 2: SSH into Cluster

```bash
ssh d23125116@147.252.6.20
```

---

## Step 3: Setup Python Environment (One-time)

### Check if phd_env exists:
```bash
ls ~/phd_env
```

### If exists, activate and verify:
```bash
source ~/phd_env/bin/activate
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
python -c "import wandb; print('W&B OK')"
```

### If any imports fail, install:
```bash
source ~/phd_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics wandb pandas numpy matplotlib seaborn scikit-learn opencv-python pillow tqdm tabulate pyyaml
```

### If phd_env doesn't exist, create it:
```bash
cd ~
python3 -m venv phd_env
source ~/phd_env/bin/activate

pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics wandb pandas numpy matplotlib seaborn scikit-learn opencv-python pillow tqdm tabulate pyyaml

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from ultralytics import YOLO; print('YOLO OK')"
```

---

## Step 4: W&B Setup

✅ **SKIP THIS STEP** - W&B key already in `configs/baseline_config.py`

The script will automatically use your W&B API key from the uploaded config file.

---

## Step 5: Create SLURM Job Scripts

### Create scripts directory:
```bash
cd ~/malaria_qgfl_experiments
mkdir -p cluster_scripts logs
```

### Create all 6 job scripts:

**Script 1: D1 + YOLOv8s**
```bash
cat > cluster_scripts/run_d1_yolov8s.sh << 'EOF'
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
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv8s on D1 (P. falciparum, 398 images)"
python -u cluster_run_baseline.py --dataset d1 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d1_yolov8s.sh
```

**Script 2: D1 + YOLOv11s**
```bash
cat > cluster_scripts/run_d1_yolov11s.sh << 'EOF'
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
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s on D1"
python -u cluster_run_baseline.py --dataset d1 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d1_yolov11s.sh
```

**Script 3: D2 + YOLOv8s**
```bash
cat > cluster_scripts/run_d2_yolov8s.sh << 'EOF'
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
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv8s on D2 (P. vivax, 1,328 images)"
python -u cluster_run_baseline.py --dataset d2 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d2_yolov8s.sh
```

**Script 4: D2 + YOLOv11s**
```bash
cat > cluster_scripts/run_d2_yolov11s.sh << 'EOF'
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
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s on D2"
python -u cluster_run_baseline.py --dataset d2 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d2_yolov11s.sh
```

**Script 5: D3 + YOLOv8s**
```bash
cat > cluster_scripts/run_d3_yolov8s.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=yolo_v8s_d3
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/yolo_v8s_d3_%j.out
#SBATCH --error=logs/yolo_v8s_d3_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=72:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv8s on D3 (Multi-species, 28,905 images)"
python -u cluster_run_baseline.py --dataset d3 --model yolov8s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d3_yolov8s.sh
```

**Script 6: D3 + YOLOv11s**
```bash
cat > cluster_scripts/run_d3_yolov11s.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=yolo_v11s_d3
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/yolo_v11s_d3_%j.out
#SBATCH --error=logs/yolo_v11s_d3_%j.err
#SBATCH --partition=MEDIUM-G2
#SBATCH --time=72:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s on D3"
python -u cluster_run_baseline.py --dataset d3 --model yolov11s --epochs 200 --batch-size 16 --use-wandb
EOF

chmod +x cluster_scripts/run_d3_yolov11s.sh
```

---

## Step 6: Submit Jobs

### Submit all 6 jobs (queue automatically):
```bash
cd ~/malaria_qgfl_experiments/cluster_scripts

sbatch run_d1_yolov8s.sh
sbatch run_d1_yolov11s.sh
sbatch run_d2_yolov8s.sh
sbatch run_d2_yolov11s.sh
sbatch run_d3_yolov8s.sh
sbatch run_d3_yolov11s.sh

# Check status
squeue -u d23125116
```

---

## Step 7: Monitor Jobs

```bash
# Check queue
squeue -u d23125116

# Watch output
tail -f ~/malaria_qgfl_experiments/logs/yolo_v8s_d1_*.out

# View all logs
ls -lt ~/malaria_qgfl_experiments/logs/
```

---

## Step 8: Download Results

### From your local machine:
```bash
rsync -avz --progress \
  d23125116@147.252.6.20:~/malaria_qgfl_experiments/results/ \
  "/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/cluster_results/"
```

---

**Ready!** Follow steps 1-6 in order. Everything is copy-paste ready.
