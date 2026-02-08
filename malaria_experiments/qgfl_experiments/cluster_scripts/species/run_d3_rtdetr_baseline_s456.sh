#!/bin/bash
#SBATCH --job-name=rtdetr_species_base_s456
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/species/rtdetr_base_s456_%j.out
#SBATCH --error=logs/species/rtdetr_base_s456_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=72:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

mkdir -p logs/species

echo "========================================"
echo "Species Baseline: RT-DETR (Seed 456)"
echo "Dataset: D3 Multi-Species (Guemas Comparison)"
echo "========================================"

python -u qgfl_experiments/cluster_run_baseline_species.py \
    --model rtdetr \
    --seed 456 \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb
