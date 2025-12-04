#!/bin/bash
#SBATCH --job-name=yolo_v11s_d1_base_s123
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v11s_d1_baseline_s123_%j.out
#SBATCH --error=logs/yolo_v11s_d1_baseline_s123_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s Baseline on D1 (P. falciparum, 398 images)"
echo "Config: seed=123"
python -u qgfl_experiments/cluster_run_baseline.py \
    --dataset d1 \
    --model yolov11s \
    --epochs 200 \
    --batch-size 16 \
    --seed 123 \
    --use-wandb
