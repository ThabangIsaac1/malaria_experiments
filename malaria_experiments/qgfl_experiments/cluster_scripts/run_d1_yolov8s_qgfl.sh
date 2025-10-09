#!/bin/bash
#SBATCH --job-name=yolo_v8s_d1_qgfl
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v8s_d1_qgfl_%j.out
#SBATCH --error=logs/yolo_v8s_d1_qgfl_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=24:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv8s + QGFL on D1 (P. falciparum, 398 images)"
python -u qgfl_experiments/cluster_run_qgfl.py --dataset d1 --model yolov8s --epochs 200 --batch-size 16 --loss-type qgfl --use-wandb
