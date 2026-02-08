#!/bin/bash
#SBATCH --job-name=yolo11s_species_base_s234
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/species/yolo11s_base_s234_%j.out
#SBATCH --error=logs/species/yolo11s_base_s234_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

mkdir -p logs/species

echo "========================================"
echo "Species Baseline: YOLOv11s (Seed 234)"
echo "Dataset: D3 Multi-Species"
echo "========================================"

python -u qgfl_experiments/cluster_run_baseline_species.py \
    --model yolov11s \
    --seed 234 \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb
