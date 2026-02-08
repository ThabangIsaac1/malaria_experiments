#!/bin/bash
#SBATCH --job-name=v11s_d3_qgfl_a035_s123
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/v11s_d3_qgfl_a035_s123_%j.out
#SBATCH --error=logs/v11s_d3_qgfl_a035_s123_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s + QGFL Alpha 0.35 on D3 (Multi-species, Seed 123)"
echo "Config: alpha=0.35 (calibrated for strong baseline)"
python -u qgfl_experiments/cluster_run_qgfl.py \
    --dataset d3 \
    --model yolov11s \
    --epochs 200 \
    --batch-size 16 \
    --loss-type qgfl \
    --alpha-infected 0.35 \
    --alpha-uninfected 0.65 \
    --gamma-infected 4.0 \
    --gamma-uninfected 2.0 \
    --difficulty-threshold 0.6 \
    --quality-margin 0.3 \
    --seed 123 \
    --use-wandb
