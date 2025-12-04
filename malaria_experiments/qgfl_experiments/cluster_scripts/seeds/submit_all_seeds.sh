#!/bin/bash
# Submit all seed experiments for statistical analysis
# Note: Cluster limit is 2 concurrent jobs - submit in batches

echo "=============================================="
echo "SEED EXPERIMENTS FOR STATISTICAL ANALYSIS"
echo "=============================================="
echo "Total: 20 runs (seeds 123 and 456)"
echo "Cluster limit: 2 concurrent jobs"
echo ""

# Priority order (submit 2 at a time, wait for completion):
# Batch 1: D2 comparisons (highest priority - weak baseline dataset)
# Batch 2: D1/D3 YOLOv11s comparisons

echo "=== BATCH 1: D2 Baseline (6 runs) ==="
echo "sbatch run_d2_yolov8s_baseline_s123.sh"
echo "sbatch run_d2_yolov8s_baseline_s456.sh"
echo "sbatch run_d2_yolov11s_baseline_s123.sh"
echo "sbatch run_d2_yolov11s_baseline_s456.sh"
echo "sbatch run_d2_rtdetr_baseline_s123.sh"
echo "sbatch run_d2_rtdetr_baseline_s456.sh"
echo ""

echo "=== BATCH 2: D2 QGFL 0.9 (6 runs) ==="
echo "sbatch run_d2_yolov8s_qgfl_s123.sh"
echo "sbatch run_d2_yolov8s_qgfl_s456.sh"
echo "sbatch run_d2_yolov11s_qgfl_s123.sh"
echo "sbatch run_d2_yolov11s_qgfl_s456.sh"
echo "sbatch run_d2_rtdetr_qgfl_s123.sh"
echo "sbatch run_d2_rtdetr_qgfl_s456.sh"
echo ""

echo "=== BATCH 3: D1 YOLOv11s (4 runs) ==="
echo "sbatch run_d1_yolov11s_baseline_s123.sh"
echo "sbatch run_d1_yolov11s_baseline_s456.sh"
echo "sbatch run_d1_yolov11s_qgfl_a035_s123.sh"
echo "sbatch run_d1_yolov11s_qgfl_a035_s456.sh"
echo ""

echo "=== BATCH 4: D3 YOLOv11s (4 runs) ==="
echo "sbatch run_d3_yolov11s_baseline_s123.sh"
echo "sbatch run_d3_yolov11s_baseline_s456.sh"
echo "sbatch run_d3_yolov11s_qgfl_s123.sh"
echo "sbatch run_d3_yolov11s_qgfl_s456.sh"
echo ""

echo "=============================================="
echo "SUBMISSION STRATEGY:"
echo "1. Submit 2 jobs at a time (cluster limit)"
echo "2. Monitor with: squeue -u d23125116"
echo "3. Check logs: tail -f logs/*.out"
echo "4. Track on W&B: malaria_qgfl_experiments"
echo "=============================================="
