#!/bin/bash
# Test script: RT-DETR baselines for D1 + D2 only (NOT D3)
# Purpose: Validate AdamW hyperparameters before full deployment

echo "=========================================="
echo "RT-DETR Baseline Test: D1 + D2 Only"
echo "=========================================="
echo ""
echo "Testing AdamW hyperparameters:"
echo "  - optimizer: AdamW"
echo "  - lr0: 0.0017"
echo "  - lrf: 0.01"
echo "  - warmup_epochs: 5"
echo "  - cls: 1.0"
echo "  - box: 7.5"
echo ""

cd ~/malaria_qgfl_experiments/qgfl_experiments/cluster_scripts

# Create logs directory if not exists
mkdir -p logs

echo "Submitting D1 RT-DETR (P. falciparum, 398 images)"
sbatch run_d1_rtdetr.sh
echo "✓ D1 submitted"
echo ""

echo "Waiting 5 seconds before D2..."
sleep 5

echo "Submitting D2 RT-DETR (P. vivax, 1,328 images)"
sbatch run_d2_rtdetr.sh
echo "✓ D2 submitted"
echo ""

echo "=========================================="
echo "2 RT-DETR test jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor queue: squeue -u d23125116"
echo "Watch D1 logs: tail -f logs/rtdetr_d1_*.out"
echo "Watch D2 logs: tail -f logs/rtdetr_d2_*.out"
echo "W&B dashboard: https://wandb.ai/learning/malaria_qgfl_experiments"
echo ""
echo "Expected runtime:"
echo "  D1: ~24 hours (398 images × 200 epochs)"
echo "  D2: ~32 hours (1,328 images × 200 epochs)"
echo ""
echo "Success criteria:"
echo "  - Max confidence ≥ 0.5"
echo "  - Test recall @ conf=0.5 ≥ 50%"
echo "  - mAP50 ≥ 60%"
echo ""
