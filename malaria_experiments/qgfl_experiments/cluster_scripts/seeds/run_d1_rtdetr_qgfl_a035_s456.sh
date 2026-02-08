#!/bin/bash
#SBATCH --job-name=rtdetr_d1_qgfl_a035_s456
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rtdetr_d1_qgfl_a035_s456_%j.out
#SBATCH --error=logs/rtdetr_d1_qgfl_a035_s456_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: RT-DETR + QGFL Alpha 0.35 on D1 (Seed 456)"
echo "Config: alpha=0.35 (calibrated for strong baseline)"
python -u qgfl_experiments/cluster_run_qgfl.py \
    --dataset d1 \
    --model rtdetr \
    --epochs 200 \
    --batch-size 16 \
    --loss-type qgfl \
    --alpha-infected 0.35 \
    --alpha-uninfected 0.65 \
    --gamma-infected 4.0 \
    --gamma-uninfected 2.0 \
    --difficulty-threshold 0.6 \
    --quality-margin 0.3 \
    --seed 456 \
    --use-wandb
