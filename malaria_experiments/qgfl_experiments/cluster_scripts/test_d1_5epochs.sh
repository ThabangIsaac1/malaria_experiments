#!/bin/bash
#SBATCH --job-name=test_5ep_d1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/test_5ep_%j.out
#SBATCH --error=logs/test_5ep_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=01:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Quick 5-epoch test: YOLOv8s on D1"
python -u qgfl_experiments/cluster_run_baseline.py --dataset d1 --model yolov8s --epochs 5 --batch-size 16 --use-wandb
