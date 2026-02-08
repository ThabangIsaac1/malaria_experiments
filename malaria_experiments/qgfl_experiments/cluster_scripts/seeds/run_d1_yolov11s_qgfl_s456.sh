#!/bin/bash
#SBATCH --job-name=v11s_d1_qgfl_s456
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/v11s_d1_qgfl_s456_%j.out
#SBATCH --error=logs/v11s_d1_qgfl_s456_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s + QGFL on D1 (P. falciparum)"
echo "Config: alpha=0.9, seed=456"
python -u qgfl_experiments/cluster_run_qgfl.py \
    --dataset d1 \
    --model yolov11s \
    --epochs 200 \
    --batch-size 16 \
    --loss-type qgfl \
    --seed 456 \
    --use-wandb
