#!/bin/bash
#SBATCH --job-name=rtdetr_saqgfl_s042
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/species/rtdetr_saqgfl_s042_%j.out
#SBATCH --error=logs/species/rtdetr_saqgfl_s042_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=48:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

mkdir -p logs/species

echo "========================================"
echo "SA-QGFL: RT-DETR (Seed 42)"
echo "Dataset: D3 Multi-Species"
echo "Components: MCP + PWA + HQS"
echo "Direct Guemas Comparison"
echo "========================================"

python -u qgfl_experiments/cluster_run_saqgfl.py \
    --model rtdetr \
    --seed 42 \
    --epochs 200 \
    --batch-size 16 \
    --use-hqs \
    --use-wandb
