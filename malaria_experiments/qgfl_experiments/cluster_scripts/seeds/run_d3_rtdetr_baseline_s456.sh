#!/bin/bash
#SBATCH --job-name=rtdetr_d3_base_s456
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rtdetr_d3_baseline_s456_%j.out
#SBATCH --error=logs/rtdetr_d3_baseline_s456_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=120:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: RT-DETR Baseline on D3 (Multi-species, seed=456)"
python -u qgfl_experiments/cluster_run_baseline.py \
    --dataset d3 \
    --model rtdetr \
    --epochs 200 \
    --batch-size 16 \
    --seed 456 \
    --use-wandb
