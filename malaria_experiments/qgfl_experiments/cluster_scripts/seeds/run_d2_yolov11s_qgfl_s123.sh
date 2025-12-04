#!/bin/bash
#SBATCH --job-name=yolo_v11s_d2_qgfl_s123
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/yolo_v11s_d2_qgfl_s123_%j.out
#SBATCH --error=logs/yolo_v11s_d2_qgfl_s123_%j.err
#SBATCH --partition=rtx8000
#SBATCH --time=32:00:00

source ~/phd_env/bin/activate
export PYTHONUNBUFFERED=1
cd ~/malaria_qgfl_experiments

echo "Job: YOLOv11s + QGFL on D2 (P. vivax, 1328 images)"
echo "Config: alpha=0.9, seed=123"
python -u qgfl_experiments/cluster_run_qgfl.py \
    --dataset d2 \
    --model yolov11s \
    --epochs 200 \
    --batch-size 16 \
    --loss-type qgfl \
    --seed 123 \
    --use-wandb
