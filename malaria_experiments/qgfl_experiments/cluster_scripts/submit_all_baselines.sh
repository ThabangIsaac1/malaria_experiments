#!/bin/bash
# Master script to submit all 9 baseline experiments
# Submits 2 jobs at a time (cluster limit)

echo "=========================================="
echo "Submitting All Baseline Experiments (9 total)"
echo "=========================================="
echo ""

cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts

# Create logs directory if not exists
mkdir -p logs

echo "Batch 1: D1 YOLO experiments (2 jobs)"
sbatch run_d1_yolov8s.sh
sbatch run_d1_yolov11s.sh
echo "✓ Submitted D1 YOLOv8s and YOLOv11s"
echo ""

echo "Waiting 30 seconds before next batch..."
sleep 30

echo "Batch 2: D1 RT-DETR + D2 YOLOv8s (2 jobs)"
sbatch run_d1_rtdetr.sh
sbatch run_d2_yolov8s.sh
echo "✓ Submitted D1 RT-DETR and D2 YOLOv8s"
echo ""

echo "Waiting 30 seconds before next batch..."
sleep 30

echo "Batch 3: D2 YOLOv11s + RT-DETR (2 jobs)"
sbatch run_d2_yolov11s.sh
sbatch run_d2_rtdetr.sh
echo "✓ Submitted D2 YOLOv11s and RT-DETR"
echo ""

echo "Waiting 30 seconds before next batch..."
sleep 30

echo "Batch 4: D3 YOLO experiments (2 jobs)"
sbatch run_d3_yolov8s.sh
sbatch run_d3_yolov11s.sh
echo "✓ Submitted D3 YOLOv8s and YOLOv11s"
echo ""

echo "Waiting 30 seconds before final job..."
sleep 30

echo "Batch 5: D3 RT-DETR (1 job)"
sbatch run_d3_rtdetr.sh
echo "✓ Submitted D3 RT-DETR"
echo ""

echo "=========================================="
echo "All 9 experiments submitted!"
echo "=========================================="
echo ""
echo "Monitor queue: squeue -u d23125116"
echo "Watch logs: tail -f logs/*.out"
echo "W&B dashboard: https://wandb.ai/learning/malaria_qgfl_experiments"
echo ""
echo "Experiment Matrix:"
echo "  D1: YOLOv8s, YOLOv11s, RT-DETR-L"
echo "  D2: YOLOv8s, YOLOv11s, RT-DETR-L"
echo "  D3: YOLOv8s, YOLOv11s, RT-DETR-L"
echo ""
echo "Parameters (all experiments):"
echo "  - Epochs: 200"
echo "  - Batch size: 16"
echo "  - Task: binary (Infected vs Uninfected)"
echo ""
