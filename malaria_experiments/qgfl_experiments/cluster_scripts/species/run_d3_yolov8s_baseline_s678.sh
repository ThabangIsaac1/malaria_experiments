#!/bin/bash
#SBATCH --job-name=yolo8s_species_base_s678
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/species/yolo8s_base_s678_%j.out
#SBATCH --error=logs/species/yolo8s_base_s678_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

mkdir -p logs/species

echo "========================================"
echo "Species Baseline: YOLOv8s (Seed 678)"
echo "Dataset: D3 Multi-Species"
echo "========================================"

python -u qgfl_experiments/cluster_run_baseline_species.py \
    --model yolov8s \
    --seed 678 \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb
