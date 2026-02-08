#!/bin/bash
# Submit all species detection experiments for Paper 2
# 54 total: 3 models × 9 seeds × 2 methods (Baseline + SA-QGFL)
#
# Seeds: 42, 123, 234, 345, 456, 567, 678, 789, 890
# Models: YOLOv8s, YOLOv11s, RT-DETR
# Methods: Baseline, SA-QGFL (MCP + PWA + HQS)
#
# Statistical Design: 9 seeds enables Wilcoxon signed-rank tests (n≥5)

echo "========================================"
echo "PAPER 2: SA-QGFL Species Detection"
echo "Submitting 54 experiments to SLURM"
echo "========================================"
echo ""
echo "Seeds: 42, 123, 234, 345, 456, 567, 678, 789, 890"
echo "Models: YOLOv8s, YOLOv11s, RT-DETR"
echo "Methods: Baseline, SA-QGFL"
echo ""

# Create logs directory
mkdir -p logs/species

# =================================================
# PHASE 1: BASELINES (27 experiments)
# =================================================
echo "Phase 1: Baselines (27 experiments)"
echo "-----------------------------------------"

# YOLOv8s Baselines (9 seeds)
echo "Submitting YOLOv8s baselines..."
sbatch run_d3_yolov8s_baseline_s042.sh
sbatch run_d3_yolov8s_baseline_s123.sh
sbatch run_d3_yolov8s_baseline_s234.sh
sbatch run_d3_yolov8s_baseline_s345.sh
sbatch run_d3_yolov8s_baseline_s456.sh
sbatch run_d3_yolov8s_baseline_s567.sh
sbatch run_d3_yolov8s_baseline_s678.sh
sbatch run_d3_yolov8s_baseline_s789.sh
sbatch run_d3_yolov8s_baseline_s890.sh

# YOLOv11s Baselines (9 seeds)
echo "Submitting YOLOv11s baselines..."
sbatch run_d3_yolov11s_baseline_s042.sh
sbatch run_d3_yolov11s_baseline_s123.sh
sbatch run_d3_yolov11s_baseline_s234.sh
sbatch run_d3_yolov11s_baseline_s345.sh
sbatch run_d3_yolov11s_baseline_s456.sh
sbatch run_d3_yolov11s_baseline_s567.sh
sbatch run_d3_yolov11s_baseline_s678.sh
sbatch run_d3_yolov11s_baseline_s789.sh
sbatch run_d3_yolov11s_baseline_s890.sh

# RT-DETR Baselines (9 seeds) - Direct Guemas comparison
echo "Submitting RT-DETR baselines (Guemas comparison)..."
sbatch run_d3_rtdetr_baseline_s042.sh
sbatch run_d3_rtdetr_baseline_s123.sh
sbatch run_d3_rtdetr_baseline_s234.sh
sbatch run_d3_rtdetr_baseline_s345.sh
sbatch run_d3_rtdetr_baseline_s456.sh
sbatch run_d3_rtdetr_baseline_s567.sh
sbatch run_d3_rtdetr_baseline_s678.sh
sbatch run_d3_rtdetr_baseline_s789.sh
sbatch run_d3_rtdetr_baseline_s890.sh

# =================================================
# PHASE 2: SA-QGFL (27 experiments)
# =================================================
echo ""
echo "Phase 2: SA-QGFL - MCP + PWA + HQS (27 experiments)"
echo "-----------------------------------------"

# YOLOv8s SA-QGFL (9 seeds)
echo "Submitting YOLOv8s SA-QGFL..."
sbatch run_d3_yolov8s_saqgfl_s042.sh
sbatch run_d3_yolov8s_saqgfl_s123.sh
sbatch run_d3_yolov8s_saqgfl_s234.sh
sbatch run_d3_yolov8s_saqgfl_s345.sh
sbatch run_d3_yolov8s_saqgfl_s456.sh
sbatch run_d3_yolov8s_saqgfl_s567.sh
sbatch run_d3_yolov8s_saqgfl_s678.sh
sbatch run_d3_yolov8s_saqgfl_s789.sh
sbatch run_d3_yolov8s_saqgfl_s890.sh

# YOLOv11s SA-QGFL (9 seeds)
echo "Submitting YOLOv11s SA-QGFL..."
sbatch run_d3_yolov11s_saqgfl_s042.sh
sbatch run_d3_yolov11s_saqgfl_s123.sh
sbatch run_d3_yolov11s_saqgfl_s234.sh
sbatch run_d3_yolov11s_saqgfl_s345.sh
sbatch run_d3_yolov11s_saqgfl_s456.sh
sbatch run_d3_yolov11s_saqgfl_s567.sh
sbatch run_d3_yolov11s_saqgfl_s678.sh
sbatch run_d3_yolov11s_saqgfl_s789.sh
sbatch run_d3_yolov11s_saqgfl_s890.sh

# RT-DETR SA-QGFL (9 seeds) - Direct Guemas comparison
echo "Submitting RT-DETR SA-QGFL (Guemas comparison)..."
sbatch run_d3_rtdetr_saqgfl_s042.sh
sbatch run_d3_rtdetr_saqgfl_s123.sh
sbatch run_d3_rtdetr_saqgfl_s234.sh
sbatch run_d3_rtdetr_saqgfl_s345.sh
sbatch run_d3_rtdetr_saqgfl_s456.sh
sbatch run_d3_rtdetr_saqgfl_s567.sh
sbatch run_d3_rtdetr_saqgfl_s678.sh
sbatch run_d3_rtdetr_saqgfl_s789.sh
sbatch run_d3_rtdetr_saqgfl_s890.sh

echo ""
echo "========================================"
echo "All 54 experiments submitted!"
echo "========================================"
echo ""
echo "Breakdown:"
echo "  - Baselines: 27 (3 models × 9 seeds)"
echo "  - SA-QGFL:   27 (3 models × 9 seeds)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Logs: logs/species/"
echo "========================================"
