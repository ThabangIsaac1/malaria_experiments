#!/bin/bash
#SBATCH --job-name=v8s_d3_qgfl_a035
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/v8s_d3_qgfl_a035_%j.out
#SBATCH --error=logs/v8s_d3_qgfl_a035_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv8s + QGFL Sweet Spot on D3"
echo "Config: alpha=0.35 (optimized for strong baseline)"
python -u qgfl_experiments/cluster_run_qgfl.py \
    --dataset d3 \
    --model yolov8s \
    --epochs 200 \
    --batch-size 16 \
    --loss-type qgfl \
    --alpha-infected 0.35 \
    --alpha-uninfected 0.65 \
    --gamma-infected 4.0 \
    --gamma-uninfected 2.0 \
    --difficulty-threshold 0.6 \
    --quality-margin 0.3 \
    --use-wandb
