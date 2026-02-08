#!/bin/bash
#SBATCH --job-name=rtdetr_species_base_s890
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/species/rtdetr_base_s890_%j.out
#SBATCH --error=logs/species/rtdetr_base_s890_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

mkdir -p logs/species

echo "========================================"
echo "Species Baseline: RT-DETR (Seed 890)"
echo "Dataset: D3 Multi-Species"
echo "Direct Guemas et al. 2024 Comparison"
echo "========================================"

python -u qgfl_experiments/cluster_run_baseline_species.py \
    --model rtdetr \
    --seed 890 \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb
