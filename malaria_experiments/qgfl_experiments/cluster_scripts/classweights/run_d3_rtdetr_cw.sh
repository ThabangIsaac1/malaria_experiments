#!/bin/bash
#SBATCH --job-name=rtdetr_d3_cw
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rtdetr_d3_cw_%j.out
#SBATCH --error=logs/rtdetr_d3_cw_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: RT-DETR + Class Weights on D3 (Multi-species, 28905 images)"
echo "Config: inverse_freq weighting, seed=42"
python -u qgfl_experiments/cluster_run_classweights.py \
    --dataset d3 \
    --model rtdetr \
    --epochs 200 \
    --batch-size 16 \
    --seed 42 \
    --use-wandb
