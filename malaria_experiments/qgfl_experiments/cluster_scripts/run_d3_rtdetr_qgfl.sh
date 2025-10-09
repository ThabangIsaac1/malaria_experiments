#!/bin/bash
#SBATCH --job-name=rtdetr_d3_qgfl
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rtdetr_d3_qgfl_%j.out
#SBATCH --error=logs/rtdetr_d3_qgfl_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=336:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: RT-DETR + QGFL on D3 (Multi-species, 28,905 images)"
python -u qgfl_experiments/cluster_run_qgfl.py --dataset d3 --model rtdetr --epochs 200 --batch-size 16 --loss-type qgfl --use-wandb
