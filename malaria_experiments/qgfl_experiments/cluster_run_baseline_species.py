#!/usr/bin/env python
# coding: utf-8
"""
Species Baseline Training Script - Multi-Species Plasmodium Detection

Standard baseline training (no SA-QGFL) for D3 dataset (5 classes: Uninfected + 4 Plasmodium species).
Uses default loss functions: BCE for YOLO, VarifocalLoss for RT-DETR.

This script provides a standalone species baseline that:
1. Uses SpeciesEvaluator for species-specific metrics
2. Logs per-species metrics to W&B
3. Computes rare species average (P. ovale, P. malariae, P. vivax)
4. Enables fair comparison with SA-QGFL experiments

Usage:
    python cluster_run_baseline_species.py --model yolov8s --seed 123
    python cluster_run_baseline_species.py --model rtdetr --seed 123

Author: Thabang Isaka
Date: 2025-12-05
"""

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================
import argparse

parser = argparse.ArgumentParser(
    description='Species Baseline Training for Multi-Species Plasmodium Detection',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Required arguments
parser.add_argument('--model', type=str, required=True,
                    choices=['yolov8s', 'yolov11s', 'rtdetr'],
                    help='Model architecture to train')

# Optional arguments (species detection defaults)
parser.add_argument('--dataset', type=str, default='d3',
                    choices=['d3'],  # Species detection is D3 only
                    help='Dataset (D3 for multi-species detection)')
parser.add_argument('--task', type=str, default='species',
                    choices=['species'],  # Species detection only
                    help='Classification task (species detection)')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs')
parser.add_argument('--batch-size', type=int, default=16,
                    help='Training batch size')
parser.add_argument('--use-wandb', action='store_true', default=True,
                    help='Enable Weights & Biases logging')
parser.add_argument('--no-wandb', action='store_true',
                    help='Disable Weights & Biases logging')
parser.add_argument('--results-dir', type=str, default='../results',
                    help='Base directory for saving results')

# Hyperparameter arguments
parser.add_argument('--optimizer', type=str, default=None,
                    choices=['auto', 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam'],
                    help='Optimizer (default: auto-select based on model)')
parser.add_argument('--lr0', type=float, default=0.01,
                    help='Initial learning rate')
parser.add_argument('--lrf', type=float, default=0.01,
                    help='Final learning rate factor')
parser.add_argument('--warmup-epochs', type=float, default=3.0,
                    help='Warmup epochs')
parser.add_argument('--cls', type=float, default=0.5,
                    help='Classification loss weight')
parser.add_argument('--box', type=float, default=7.5,
                    help='Box loss weight')

# Seed for reproducibility
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (use 42, 123, 234, ... for multi-seed runs)')

# Mini dataset for smoke testing
parser.add_argument('--mini', action='store_true',
                    help='Use mini 50-image dataset for smoke testing')

args = parser.parse_args()

# Handle W&B flag
if args.no_wandb:
    args.use_wandb = False

# Auto-select optimizer
if args.optimizer is None:
    if 'rtdetr' in args.model.lower():
        args.optimizer = 'auto'
    else:
        args.optimizer = 'SGD'

print("=" * 70)
print("SPECIES BASELINE TRAINING")
print("Multi-Species Plasmodium Detection (D3 Dataset)")
print("=" * 70)
print(f"Dataset: {args.dataset.upper()} (Multi-Species)")
print(f"Model: {args.model}")
print(f"Task: {args.task}")
print(f"Loss Type: Default (BCE for YOLO, VFL for RT-DETR)")
print(f"Seed: {args.seed}")
print(f"Epochs: {args.epochs}")
print(f"Batch Size: {args.batch_size}")
print(f"W&B Logging: {args.use_wandb}")
print("=" * 70)

# ============================================================================
# SETUP AND IMPORTS
# ============================================================================
import sys
from pathlib import Path
import json

# Get script directory and add to path
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tabulate import tabulate
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import torch
import yaml
import os
import socket
import random
from src.utils.visualizer import YOLOVisualizer
import cv2
from scipy.ndimage import gaussian_filter
from collections import defaultdict

# Set figure quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# Import species evaluator
from src.evaluation.evaluator_species import SpeciesEvaluator

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
print(f"\n[SEED] Random seed set to: {args.seed}")

print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

# ============================================================================
# CONFIGURATION
# ============================================================================
from configs.baseline_config import ExperimentConfig

config = ExperimentConfig(
    dataset=args.dataset,
    task=args.task,
    model_name=args.model,
    epochs=args.epochs,
    batch_size=args.batch_size,
    use_wandb=args.use_wandb
)

experiment_name = config.get_experiment_name()
print(f"\nExperiment: {experiment_name}")
print(f"Dataset: {config.dataset.upper()}")
print(f"Task: {config.task}")
print(f"Model: {config.model_name}")

# ============================================================================
# DATASET PREPARATION
# ============================================================================
from src.utils.paths import get_dataset_paths, verify_dataset

dataset_valid = verify_dataset(config.dataset, config.task)
if not dataset_valid:
    raise RuntimeError(f"Dataset preparation failed for {config.dataset}/{config.task}")

# Use mini dataset for smoke testing if --mini flag is set
if args.mini:
    print("\n*** SMOKE TEST MODE: Using mini 50-image dataset ***")
    yolo_path = Path("/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments/dataset_d3_mini/yolo_format/species")
    yaml_suffix = "_mini"
else:
    yolo_path = get_dataset_paths(config.dataset, config.task)
    yaml_suffix = ""

# Create YAML
yaml_dir = script_dir / 'configs' / 'data_yamls'
yaml_dir.mkdir(parents=True, exist_ok=True)
yaml_path = yaml_dir / f'{config.dataset}_{config.task}{yaml_suffix}.yaml'

data_yaml = {
    'path': str(yolo_path),
    'train': 'train/images',
    'val': 'val/images',
    'test': 'test/images',
    'names': config.get_class_names(),
    'nc': len(config.get_class_names())
}

with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Created YAML: {yaml_path}")
print(f"Number of classes: {data_yaml['nc']}")
print(f"Class names: {data_yaml['names']}")

# ============================================================================
# DATASET STATISTICS
# ============================================================================
visualizer = YOLOVisualizer(config.get_class_names())

print("\n" + "=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

stats = visualizer.get_dataset_stats(yolo_path)

total_images = sum(s['images'] for s in stats.values())
total_boxes = sum(s['total_boxes'] for s in stats.values())

print(f"\nOVERALL:")
print(f"  Total images: {total_images}")
print(f"  Total annotations: {total_boxes}")

total_per_class = {}
for s in stats.values():
    for class_id, count in s['class_distribution'].items():
        total_per_class[class_id] = total_per_class.get(class_id, 0) + count

print(f"\n  Class Distribution:")
for class_id in sorted(total_per_class.keys()):
    count = total_per_class[class_id]
    class_name = config.get_class_names().get(class_id)
    percentage = (count/total_boxes*100) if total_boxes > 0 else 0
    print(f"    {class_name}: {count:,} ({percentage:.1f}%)")

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
from ultralytics import YOLO, RTDETR

if config.model_name == 'yolov8s':
    model = YOLO('yolov8s.pt')
elif config.model_name in ['yolo11s', 'yolov11s']:
    model = YOLO('yolo11s.pt')
elif config.model_name == 'rtdetr':
    model = RTDETR('rtdetr-l.pt')
else:
    raise ValueError(f"Model {config.model_name} not implemented")

print(f"\nModel loaded: {config.model_name}")

# ============================================================================
# WANDB INITIALIZATION
# ============================================================================
import wandb
import time

if args.use_wandb:
    # Species baseline project name (same as SA-QGFL for comparison)
    wandb_project = "malaria_qgfl_multispecies_experiments"

    # Environment detection (cluster vs laptop)
    is_cluster = (
        os.getenv('SLURM_JOB_ID') is not None or
        os.getenv('PBS_JOBID') is not None or
        'hpc' in socket.gethostname().lower() or
        'node' in socket.gethostname().lower()
    )
    env_prefix = "cluster_" if is_cluster else "laptop_"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"{env_prefix}baseline_{config.model_name}_{config.dataset}_s{args.seed:03d}_{timestamp}"

    # Comprehensive W&B config
    wandb_config = {
        # Model
        'model': config.model_name,
        'dataset': config.dataset,
        'task': config.task,

        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'warmup_epochs': args.warmup_epochs,
        'cls_weight': args.cls,
        'box_weight': args.box,

        # Loss type (baseline - no SA-QGFL)
        'loss_type': 'default',  # BCE for YOLO, VFL for RT-DETR

        # Evaluation thresholds (Guemas methodology)
        'evaluation/conf_threshold': config.conf,
        'evaluation/iou_threshold': config.iou,
        'evaluation/agnostic_nms': True,

        # Dataset info
        'num_classes': len(config.get_class_names()),
        'class_names': list(config.get_class_names().values()),

        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'environment': 'cluster' if is_cluster else 'laptop',
        'hostname': socket.gethostname(),
    }

    wandb.init(
        project=wandb_project,
        name=run_name,
        config=wandb_config,
        tags=['baseline', 'species', config.model_name, config.dataset, f's{args.seed}']
    )

    # Define custom metrics for proper x-axis tracking
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("metrics/*", step_metric="epoch")
    wandb.define_metric("lr/*", step_metric="epoch")

    print(f"\n[W&B] Initialized: {wandb_project}/{run_name}")
    print(f"[W&B] Environment: {'CLUSTER' if is_cluster else 'LAPTOP'}")
    print(f"[W&B] View run at: {wandb.run.url}")

# ============================================================================
# TRAINING
# ============================================================================
print("\n" + "=" * 70)
print("STARTING BASELINE TRAINING")
print("=" * 70)

# Results directory
results_base = Path(args.results_dir) / 'baseline' / config.dataset / config.task
run_dir = results_base / f'{config.model_name}_s{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
run_dir.mkdir(parents=True, exist_ok=True)

print(f"Results will be saved to: {run_dir}")

# Training hyperparameters
train_args = {
    'data': str(yaml_path),
    'epochs': args.epochs,
    'batch': args.batch_size,
    'imgsz': 640,
    'project': str(run_dir),
    'name': 'train',
    'exist_ok': True,
    'pretrained': True,
    'optimizer': args.optimizer,
    'lr0': args.lr0,
    'lrf': args.lrf,
    'warmup_epochs': args.warmup_epochs,
    'cls': args.cls,
    'box': args.box,
    'seed': args.seed,
    'patience': 50,
    'save': True,
    'plots': True,
    'cache': False,
    'workers': 8,
    'verbose': True,
}

# RT-DETR specific settings
if 'rtdetr' in config.model_name.lower():
    train_args['lr0'] = 0.0001  # Lower LR for transformer
    train_args['lrf'] = 0.01

print(f"\nTraining configuration:")
for key, value in train_args.items():
    print(f"  {key}: {value}")

# Start training
start_time = time.time()
results = model.train(**train_args)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time/3600:.2f} hours")

# ============================================================================
# ANALYSIS FUNCTIONS (Ported from binary QGFL for species detection)
# ============================================================================

# Species-specific class configuration
SPECIES_CLASS_NAMES = {
    0: 'Uninfected',
    1: 'P_falciparum',
    2: 'P_ovale',
    3: 'P_malariae',
    4: 'P_vivax',
}
SPECIES_COLORS = {
    0: (0, 255, 0),      # Green - Uninfected
    1: (255, 0, 0),      # Red - P. falciparum
    2: (255, 165, 0),    # Orange - P. ovale (rare)
    3: (128, 0, 128),    # Purple - P. malariae (rare)
    4: (0, 191, 255),    # DeepSkyBlue - P. vivax (rare)
}
RARE_SPECIES_IDS = [2, 3, 4]  # P_ovale, P_malariae, P_vivax
ALL_PARASITE_IDS = [1, 2, 3, 4]  # All species except uninfected

# Confidence thresholds (Guemas methodology)
CONF_HIGH = 0.25
CONF_LOW = 0.15


def calculate_iou_species(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    x1_min, y1_min, w1, h1 = box1
    x2_min, y2_min, w2, h2 = box2

    x1_max = x1_min + w1
    y1_max = y1_min + h1
    x2_max = x2_min + w2
    y2_max = y2_min + h2

    intersect_xmin = max(x1_min, x2_min)
    intersect_ymin = max(y1_min, y2_min)
    intersect_xmax = min(x1_max, x2_max)
    intersect_ymax = min(y1_max, y2_max)

    if intersect_xmax < intersect_xmin or intersect_ymax < intersect_ymin:
        return 0.0

    intersect_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersect_area

    return intersect_area / union_area if union_area > 0 else 0


def calculate_proper_metrics_species(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Calculate TP/FP/FN using IoU matching for species detection"""
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx not in matched_gt:
                iou = calculate_iou_species(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def analyze_model_decisions_species(model, test_images, model_name, save_dir, conf_threshold=0.25, iou_threshold=0.45, num_samples_to_display=6):
    """
    Analyze model decisions for multi-species Plasmodium detection.

    Enhanced 6-panel visualization matching binary QGFL reference:
    - Panel 1: Ground Truth (species-specific colors, solid lines)
    - Panel 2: Predictions (conf >= CONF_HIGH)
    - Panel 3: Low Confidence ([CONF_LOW, CONF_HIGH) uncertain range)
    - Panel 4-5: Decision Heatmap with uncertainty contours
    - Panel 6: Cell Examples Montage (bottom row)

    Creates decision_analysis_complete.csv for W&B table logging.
    """
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    # Species-specific matplotlib colors for GT panel
    SPECIES_COLORS_MPL = {
        0: 'green',       # Uninfected
        1: 'red',         # P. falciparum
        2: 'orange',      # P. ovale (rare)
        3: 'purple',      # P. malariae (rare)
        4: 'deepskyblue', # P. vivax (rare)
    }

    # Create output directory
    decision_dir = save_dir / model_name / 'decision_analysis'
    decision_dir.mkdir(parents=True, exist_ok=True)

    csv_data = []
    max_viz_to_save = 200  # Limit visualizations to save space
    all_confidence_data = {cls: [] for cls in range(5)}

    print(f"Analyzing {len(test_images)} test images for species detection...")
    print(f"Will display {num_samples_to_display} in output")
    print(f"Saving up to {max_viz_to_save} visualizations")
    print(f"Confidence thresholds: Confident≥{CONF_HIGH}, Uncertain=[{CONF_LOW},{CONF_HIGH})")

    for img_idx, img_path in enumerate(tqdm(test_images, desc="Decision analysis")):
        img_id = f"{img_path.stem[:12]}_{img_idx:03d}"

        img = Image.open(img_path)
        img_array = np.array(img)
        height, width = img_array.shape[:2]

        # Get ground truth per species
        label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
        gt_per_species = defaultdict(int)
        gt_boxes_per_species = defaultdict(list)

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = (cx - w/2) * width
                        y1 = (cy - h/2) * height
                        box_w = w * width
                        box_h = h * height

                        gt_per_species[cls_id] += 1
                        gt_boxes_per_species[cls_id].append([x1, y1, box_w, box_h])

        # Get predictions at CONF_HIGH and CONF_LOW thresholds
        results_normal = model.predict(img_path, conf=CONF_HIGH, iou=iou_threshold, agnostic_nms=True, verbose=False)[0]
        results_low = model.predict(img_path, conf=CONF_LOW, iou=iou_threshold, agnostic_nms=True, verbose=False)[0]

        pred_per_species = defaultdict(int)
        pred_boxes_per_species = defaultdict(list)
        pred_conf_per_species = defaultdict(list)

        if results_normal.boxes is not None:
            for box in results_normal.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                pred_per_species[cls] += 1
                pred_boxes_per_species[cls].append([x1, y1, x2-x1, y2-y1])
                pred_conf_per_species[cls].append(conf)
                all_confidence_data[cls].append(conf)

        # Count low confidence detections
        low_conf_count = 0
        low_conf_data = []
        if results_low.boxes is not None:
            for box in results_low.boxes:
                conf = box.conf.item()
                if CONF_LOW <= conf < CONF_HIGH:
                    low_conf_count += 1
                    cls = int(box.cls.item())
                    low_conf_data.append(f"{SPECIES_CLASS_NAMES[cls]}:{conf:.3f}")

        # Calculate metrics per species
        metrics_per_species = {}
        for cls_id in range(5):
            cls_name = SPECIES_CLASS_NAMES[cls_id]
            gt_boxes = gt_boxes_per_species[cls_id]
            pred_boxes = pred_boxes_per_species[cls_id]
            confs = pred_conf_per_species[cls_id]

            tp, fp, fn = calculate_proper_metrics_species(gt_boxes, pred_boxes, iou_threshold)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_per_species[cls_name] = {
                'gt': gt_per_species[cls_id],
                'pred': pred_per_species[cls_id],
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': precision, 'recall': recall, 'f1': f1,
                'conf_mean': np.mean(confs) if confs else 0,
                'conf_std': np.std(confs) if confs else 0,
                'conf_min': min(confs) if confs else 0,
                'conf_max': max(confs) if confs else 0,
            }

        # Aggregate parasite metrics (all species combined)
        parasite_tp = sum(metrics_per_species[SPECIES_CLASS_NAMES[c]]['tp'] for c in ALL_PARASITE_IDS)
        parasite_fp = sum(metrics_per_species[SPECIES_CLASS_NAMES[c]]['fp'] for c in ALL_PARASITE_IDS)
        parasite_fn = sum(metrics_per_species[SPECIES_CLASS_NAMES[c]]['fn'] for c in ALL_PARASITE_IDS)
        parasite_gt = sum(gt_per_species[c] for c in ALL_PARASITE_IDS)

        parasite_precision = parasite_tp / (parasite_tp + parasite_fp) if (parasite_tp + parasite_fp) > 0 else 0
        parasite_recall = parasite_tp / (parasite_tp + parasite_fn) if (parasite_tp + parasite_fn) > 0 else 0
        parasite_f1 = 2 * (parasite_precision * parasite_recall) / (parasite_precision + parasite_recall) if (parasite_precision + parasite_recall) > 0 else 0

        # Rare species metrics
        rare_tp = sum(metrics_per_species[SPECIES_CLASS_NAMES[c]]['tp'] for c in RARE_SPECIES_IDS)
        rare_fn = sum(metrics_per_species[SPECIES_CLASS_NAMES[c]]['fn'] for c in RARE_SPECIES_IDS)
        rare_gt = sum(gt_per_species[c] for c in RARE_SPECIES_IDS)
        rare_recall = rare_tp / (rare_tp + rare_fn) if (rare_tp + rare_fn) > 0 else 0

        # Build CSV row with enhanced columns
        csv_row = {
            'Image_ID': img_id,
            'Image_Path': img_path.name,
            'Model': model_name,
            'Width': width,
            'Height': height,
            # Ground truth counts
            'GT_Uninfected': gt_per_species[0],
            'GT_P_falciparum': gt_per_species[1],
            'GT_P_ovale': gt_per_species[2],
            'GT_P_malariae': gt_per_species[3],
            'GT_P_vivax': gt_per_species[4],
            'GT_Total_Parasites': parasite_gt,
            'GT_Total_Rare': rare_gt,
            # Prediction counts
            'Pred_Uninfected': pred_per_species[0],
            'Pred_P_falciparum': pred_per_species[1],
            'Pred_P_ovale': pred_per_species[2],
            'Pred_P_malariae': pred_per_species[3],
            'Pred_P_vivax': pred_per_species[4],
            # Performance
            'TP_Parasites': parasite_tp,
            'FP_Parasites': parasite_fp,
            'FN_Parasites': parasite_fn,
            'Parasite_Recall': parasite_recall,
            'Parasite_Precision': parasite_precision,
            'Parasite_F1': parasite_f1,
            'Rare_Species_Recall': rare_recall,
            # Per-species recall
            'P_falciparum_Recall': metrics_per_species['P_falciparum']['recall'],
            'P_ovale_Recall': metrics_per_species['P_ovale']['recall'],
            'P_malariae_Recall': metrics_per_species['P_malariae']['recall'],
            'P_vivax_Recall': metrics_per_species['P_vivax']['recall'],
            # Confidence stats (enhanced)
            'P_falciparum_Conf_Mean': metrics_per_species['P_falciparum']['conf_mean'],
            'P_falciparum_Conf_Std': metrics_per_species['P_falciparum']['conf_std'],
            'P_falciparum_Conf_Min': metrics_per_species['P_falciparum']['conf_min'],
            'P_falciparum_Conf_Max': metrics_per_species['P_falciparum']['conf_max'],
            'P_ovale_Conf_Mean': metrics_per_species['P_ovale']['conf_mean'],
            'P_malariae_Conf_Mean': metrics_per_species['P_malariae']['conf_mean'],
            'P_vivax_Conf_Mean': metrics_per_species['P_vivax']['conf_mean'],
            # Low confidence analysis (NEW)
            'Low_Conf_Count': low_conf_count,
            'Low_Conf_Details': '|'.join(low_conf_data) if low_conf_data else '',
            # Decision quality
            'Decision_Quality': 'Good' if parasite_recall > 0.7 else 'Poor',
            'Visualization_Path': f'decision_{img_id}.png'
        }
        csv_data.append(csv_row)

        # Create 6-panel visualization (matching binary QGFL reference)
        if img_idx < max_viz_to_save:
            fig = plt.figure(figsize=(24, 8))
            gs = fig.add_gridspec(2, 6, hspace=0.15, wspace=0.2, height_ratios=[3, 1])

            fig.suptitle(f'Species Decision Analysis - {model_name} - {img_id}', fontsize=16, fontweight='bold')

            # Panel 1: Ground Truth (species-specific colors, solid lines)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.imshow(img_array)

            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            x1 = int((cx - w/2) * width)
                            y1 = int((cy - h/2) * height)
                            box_w = int(w * width)
                            box_h = int(h * height)

                            color = SPECIES_COLORS_MPL[cls_id]
                            rect = Rectangle((x1, y1), box_w, box_h,
                                            linewidth=2, edgecolor=color,
                                            facecolor='none', linestyle='-')
                            ax1.add_patch(rect)

            gt_str = f"U:{gt_per_species[0]} Pf:{gt_per_species[1]} Po:{gt_per_species[2]} Pm:{gt_per_species[3]} Pv:{gt_per_species[4]}"
            ax1.set_title(f'Ground Truth (Solid Lines)\n{gt_str}', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Panel 2: Predictions conf >= CONF_HIGH
            ax2 = fig.add_subplot(gs[0, 2])
            viz_pred = img_array.copy()

            if results_normal.boxes is not None:
                for box in results_normal.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()

                    color = SPECIES_COLORS.get(cls, (128, 128, 128))
                    cv2.rectangle(viz_pred, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(viz_pred, f'P:{conf:.2f}', (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)

            ax2.imshow(viz_pred)
            pred_total = sum(pred_per_species.values())
            ax2.set_title(f'Predictions (Conf≥{CONF_HIGH})\n(Total: {pred_total})', fontsize=11)
            ax2.axis('off')

            # Panel 3: Low Confidence [CONF_LOW, CONF_HIGH)
            ax3 = fig.add_subplot(gs[0, 3])
            viz_low = img_array.copy()

            additional_detections = 0
            if results_low.boxes is not None:
                for box in results_low.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()

                    if CONF_LOW <= conf < CONF_HIGH:
                        # Dimmer colors for uncertain
                        base_color = SPECIES_COLORS.get(cls, (128, 128, 128))
                        dim_color = tuple(c // 2 for c in base_color)
                        cv2.rectangle(viz_low, (x1, y1), (x2, y2), dim_color, 1)
                        cv2.putText(viz_low, f'{conf:.2f}', (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, dim_color, 1)
                        additional_detections += 1

            ax3.imshow(viz_low)
            ax3.set_title(f'Uncertain [{CONF_LOW}-{CONF_HIGH})\n+{additional_detections} uncertain', fontsize=11)
            ax3.axis('off')

            # Panel 4-5: Decision Heatmap with Uncertainty Contours
            ax4 = fig.add_subplot(gs[0, 4:])

            decision_map = np.zeros((height, width), dtype=float)
            uncertainty_map = np.zeros((height, width), dtype=float)

            if results_low.boxes is not None:
                for box in results_low.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = box.conf.item()
                    cls = int(box.cls.item())

                    # Clamp coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    if cls in ALL_PARASITE_IDS:  # Infected - positive values (RED)
                        decision_map[y1:y2, x1:x2] = np.maximum(
                            decision_map[y1:y2, x1:x2], conf)
                    else:  # Uninfected - negative values (GREEN)
                        decision_map[y1:y2, x1:x2] = np.minimum(
                            decision_map[y1:y2, x1:x2], -conf)

                    # Uncertainty for low confidence
                    if CONF_LOW < conf < 0.40:
                        uncertainty_map[y1:y2, x1:x2] = 1 - abs(conf - 0.20) * 4

            # Smooth maps
            decision_smooth = gaussian_filter(decision_map, sigma=5)

            # Create custom colormap: Green (negative) to White (0) to Red (positive)
            colors_map = [(0, 0.5, 0), (1, 1, 1), (1, 0, 0)]  # Green -> White -> Red
            cmap = LinearSegmentedColormap.from_list('GWR', colors_map, N=100)

            ax4.imshow(img_array)
            im = ax4.imshow(decision_smooth, cmap=cmap, alpha=0.6, vmin=-1, vmax=1)

            # Add uncertainty overlay (YELLOW CONTOURS)
            uncertainty_smooth = gaussian_filter(uncertainty_map, sigma=3)
            if np.max(uncertainty_smooth) > 0.1:
                ax4.contour(uncertainty_smooth, levels=[0.5], colors='yellow', linewidths=2)

            ax4.set_title('Decision Confidence Map\n(Red: Parasites, Green: Uninfected, Yellow: Uncertain)', fontsize=11)
            ax4.axis('off')

            # Panel 5-6 (Bottom Row): Cell Examples Montage
            examples = {'parasites': [], 'uninfected': [], 'uncertain': []}

            if results_low.boxes is not None:
                for box in results_low.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls = int(box.cls.item())
                    conf = box.conf.item()

                    # Clamp and extract crop
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(width, x2), min(height, y2)
                    crop = img_array[y1c:y2c, x1c:x2c]

                    if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                        crop_resized = cv2.resize(crop, (64, 64))

                        if CONF_LOW <= conf < CONF_HIGH:
                            examples['uncertain'].append((crop_resized, conf, cls))
                        elif cls in ALL_PARASITE_IDS:
                            examples['parasites'].append((crop_resized, conf, cls))
                        else:
                            examples['uninfected'].append((crop_resized, conf, cls))

            # Display examples in bottom row
            for idx, (category, cells) in enumerate(examples.items()):
                ax = fig.add_subplot(gs[1, idx*2:idx*2+2])

                if cells:
                    cells_sorted = sorted(cells, key=lambda x: x[1], reverse=True)[:3]

                    if len(cells_sorted) >= 3:
                        montage = np.hstack([c[0] for c in cells_sorted])
                    elif len(cells_sorted) == 2:
                        montage = np.hstack([c[0] for c in cells_sorted])
                    else:
                        montage = cells_sorted[0][0]

                    ax.imshow(montage)
                    species_names = [SPECIES_CLASS_NAMES.get(c[2], '?') for c in cells_sorted]
                    ax.set_title(f'{category.capitalize()} Examples\n' +
                               f'Conf: {", ".join([f"{c[1]:.2f}" for c in cells_sorted])}',
                               fontsize=10)
                else:
                    ax.text(0.5, 0.5, f'No {category}\ndetections',
                           ha='center', va='center', fontsize=11,
                           transform=ax.transAxes)
                ax.axis('off')

            # Save visualization
            fig_path = decision_dir / f'decision_{img_id}.png'
            fig.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')

            if img_idx < num_samples_to_display:
                plt.show()
            plt.close(fig)

        if (img_idx + 1) % 20 == 0:
            print(f"  Processed {img_idx + 1}/{len(test_images)} images...")

    # Save CSV
    csv_df = pd.DataFrame(csv_data)
    csv_path = decision_dir / 'decision_analysis_complete.csv'
    csv_df.to_csv(csv_path, index=False)

    # Save Excel
    excel_path = decision_dir / 'decision_analysis_complete.xlsx'
    csv_df.to_excel(excel_path, index=False)

    # Save summary
    summary = {
        'Model': model_name,
        'Total_Images': len(test_images),
        'Mean_Parasite_Recall': float(csv_df['Parasite_Recall'].mean()),
        'Mean_Parasite_F1': float(csv_df['Parasite_F1'].mean()),
        'Mean_Rare_Species_Recall': float(csv_df['Rare_Species_Recall'].mean()),
        'Mean_P_falciparum_Recall': float(csv_df['P_falciparum_Recall'].mean()),
        'Mean_P_ovale_Recall': float(csv_df['P_ovale_Recall'].mean()),
        'Mean_P_malariae_Recall': float(csv_df['P_malariae_Recall'].mean()),
        'Mean_P_vivax_Recall': float(csv_df['P_vivax_Recall'].mean()),
        'Mean_Low_Conf_Count': float(csv_df['Low_Conf_Count'].mean()),
    }

    json_path = decision_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SPECIES DECISION ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Images analyzed: {len(test_images)}")
    print(f"✓ CSV saved: {csv_path}")
    print(f"✓ Summary saved: {json_path}")
    print(f"✓ Mean Parasite Recall: {summary['Mean_Parasite_Recall']:.3f}")
    print(f"✓ Mean Rare Species Recall: {summary['Mean_Rare_Species_Recall']:.3f}")
    print(f"✓ Mean Low Conf Detections: {summary['Mean_Low_Conf_Count']:.1f}")

    return csv_df, summary


def compute_species_recall_variability(model, yolo_path, config, experiment_name, results_dir):
    """
    Compute per-image recall variability for species detection.

    For each image, computes:
    - Per-species recall (P_falciparum, P_ovale, P_malariae, P_vivax)
    - Rare species combined recall
    - Overall parasite recall

    Creates recall_variability_data.csv and visualization.
    """
    print("\n" + "="*70)
    print("SPECIES RECALL VARIABILITY ANALYSIS")
    print("="*70)

    img_dir = yolo_path / 'test' / 'images'
    lbl_dir = yolo_path / 'test' / 'labels'

    variability_data = []

    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

    for img_path in tqdm(img_files, desc="Analyzing recall variability"):
        label_path = lbl_dir / (img_path.stem + '.txt')

        if not label_path.exists():
            continue

        img = Image.open(img_path)
        img_width, img_height = img.size

        # Count GT per species
        gt_per_species = defaultdict(int)
        gt_boxes_per_species = defaultdict(list)

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    gt_per_species[cls_id] += 1

                    cx = float(parts[1]) * img_width
                    cy = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height
                    gt_boxes_per_species[cls_id].append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

        # Skip if no parasites
        total_parasites = sum(gt_per_species[c] for c in ALL_PARASITE_IDS)
        if total_parasites == 0:
            continue

        total_cells = sum(gt_per_species.values())
        parasitemia = (total_parasites / total_cells) * 100 if total_cells > 0 else 0

        # Get predictions
        results = model.predict(img_path, conf=config.conf, iou=config.iou, agnostic_nms=True, verbose=False)[0]

        # Calculate recall per species
        species_tp = defaultdict(int)

        if results.boxes is not None:
            for box in results.boxes:
                pred_class = int(box.cls.item())
                if pred_class in ALL_PARASITE_IDS:
                    pred_box = box.xyxy[0].tolist()

                    for gt_box in gt_boxes_per_species[pred_class]:
                        # Simple IoU check
                        x1 = max(pred_box[0], gt_box[0])
                        y1 = max(pred_box[1], gt_box[1])
                        x2 = min(pred_box[2], gt_box[2])
                        y2 = min(pred_box[3], gt_box[3])

                        if x2 > x1 and y2 > y1:
                            inter = (x2 - x1) * (y2 - y1)
                            area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                            area2 = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                            iou = inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0

                            if iou > config.iou:
                                species_tp[pred_class] += 1
                                break

        # Calculate recalls
        pf_recall = species_tp[1] / gt_per_species[1] if gt_per_species[1] > 0 else np.nan
        po_recall = species_tp[2] / gt_per_species[2] if gt_per_species[2] > 0 else np.nan
        pm_recall = species_tp[3] / gt_per_species[3] if gt_per_species[3] > 0 else np.nan
        pv_recall = species_tp[4] / gt_per_species[4] if gt_per_species[4] > 0 else np.nan

        rare_gt = sum(gt_per_species[c] for c in RARE_SPECIES_IDS)
        rare_tp = sum(species_tp[c] for c in RARE_SPECIES_IDS)
        rare_recall = rare_tp / rare_gt if rare_gt > 0 else np.nan

        parasite_recall = sum(species_tp.values()) / total_parasites if total_parasites > 0 else 0

        variability_data.append({
            'image_id': img_path.stem,
            'parasitemia_percent': parasitemia,
            'total_parasites': total_parasites,
            'total_rare': rare_gt,
            'parasite_recall': parasite_recall,
            'rare_species_recall': rare_recall if not np.isnan(rare_recall) else 0,
            'P_falciparum_recall': pf_recall if not np.isnan(pf_recall) else 0,
            'P_ovale_recall': po_recall if not np.isnan(po_recall) else 0,
            'P_malariae_recall': pm_recall if not np.isnan(pm_recall) else 0,
            'P_vivax_recall': pv_recall if not np.isnan(pv_recall) else 0,
        })

    # Create DataFrame
    recall_df = pd.DataFrame(variability_data)

    # Save CSV
    save_dir = results_dir / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / 'recall_variability_data.csv'
    recall_df.to_csv(csv_path, index=False)
    print(f"Saved recall variability data to: {csv_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Species Recall Variability Analysis', fontsize=14, fontweight='bold')

    # 1. Overall parasite recall vs parasitemia
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(recall_df['parasitemia_percent'], recall_df['parasite_recall'],
                          c=recall_df['parasite_recall'], cmap='RdYlGn',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Parasitemia (%)', fontsize=11)
    ax1.set_ylabel('Parasite Recall', fontsize=11)
    ax1.set_title('Overall Parasite Detection vs Parasitemia', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Recall')

    # 2. Rare species recall distribution
    ax2 = axes[0, 1]
    rare_recalls = recall_df[recall_df['total_rare'] > 0]['rare_species_recall']
    if len(rare_recalls) > 0:
        ax2.hist(rare_recalls, bins=20, color='purple', alpha=0.7, edgecolor='black')
        ax2.axvline(rare_recalls.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rare_recalls.mean():.3f}')
    ax2.set_xlabel('Rare Species Recall', fontsize=11)
    ax2.set_ylabel('Number of Images', fontsize=11)
    ax2.set_title('Rare Species Recall Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Per-species recall boxplot
    ax3 = axes[1, 0]
    species_recalls = []
    species_labels = []
    for species in ['P_falciparum_recall', 'P_ovale_recall', 'P_malariae_recall', 'P_vivax_recall']:
        valid_data = recall_df[recall_df[species] > 0][species].dropna()
        if len(valid_data) > 0:
            species_recalls.append(valid_data)
            species_labels.append(species.replace('_recall', '').replace('_', '. '))

    if species_recalls:
        bp = ax3.boxplot(species_recalls, labels=species_labels, patch_artist=True)
        colors = ['red', 'orange', 'purple', 'deepskyblue']
        for patch, color in zip(bp['boxes'], colors[:len(species_recalls)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax3.set_ylabel('Recall', fontsize=11)
    ax3.set_title('Per-Species Recall Distribution', fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # 4. Recall vs parasitemia by density bin
    ax4 = axes[1, 1]
    bins = [(0, 1, '0-1%'), (1, 3, '1-3%'), (3, 5, '3-5%'), (5, 100, '>5%')]
    bin_means = []
    bin_stds = []
    bin_labels = []

    for low, high, label in bins:
        mask = (recall_df['parasitemia_percent'] >= low) & (recall_df['parasitemia_percent'] < high)
        bin_data = recall_df[mask]['parasite_recall']
        if len(bin_data) > 0:
            bin_means.append(bin_data.mean())
            bin_stds.append(bin_data.std())
            bin_labels.append(f'{label}\n(n={len(bin_data)})')
        else:
            bin_means.append(0)
            bin_stds.append(0)
            bin_labels.append(f'{label}\n(n=0)')

    colors = ['#B71C1C', '#FF6F00', '#FDD835', '#43A047']
    bars = ax4.bar(range(len(bin_labels)), bin_means, yerr=bin_stds, capsize=5,
                   color=colors, edgecolor='black', alpha=0.8)
    ax4.set_xticks(range(len(bin_labels)))
    ax4.set_xticklabels(bin_labels)
    ax4.set_ylabel('Mean Parasite Recall', fontsize=11)
    ax4.set_title('Recall by Parasitemia Level', fontsize=12)
    ax4.set_ylim(0, 1.05)
    ax4.axhline(0.8, color='green', linestyle=':', linewidth=1.5, label='Target: 0.8')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    viz_path = save_dir / 'recall_variability.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nRecall Variability Summary:")
    print(f"  Images analyzed: {len(recall_df)}")
    print(f"  Mean Parasite Recall: {recall_df['parasite_recall'].mean():.3f}")
    if len(rare_recalls) > 0:
        print(f"  Mean Rare Species Recall: {rare_recalls.mean():.3f}")

    return recall_df, {
        'wandb_table': recall_df.to_dict('records'),
        'summary': {
            'n_images': len(recall_df),
            'mean_parasite_recall': float(recall_df['parasite_recall'].mean()),
            'std_parasite_recall': float(recall_df['parasite_recall'].std()),
        }
    }


def plot_species_stratified_analysis(test_results, experiment_name, save_dir):
    """Create stratified analysis visualization for species detection"""

    stratified = test_results.get('stratified', {})
    rare_stratified = test_results.get('rare_species_stratified', {})

    if not stratified and not rare_stratified:
        print("No stratified data available for visualization")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Stratified Performance Analysis - Species Detection', fontsize=14, fontweight='bold')

    bins = ['0-1%', '1-3%', '3-5%', '>5%']
    colors = ['#B71C1C', '#FF6F00', '#FDD835', '#43A047']

    # Panel 1: Total Parasite Stratified
    ax1 = axes[0]
    if stratified:
        recalls = [stratified.get(b, {}).get('mean_recall', 0) for b in bins]
        stds = [stratified.get(b, {}).get('std_recall', 0) for b in bins]
        counts = [stratified.get(b, {}).get('count', 0) for b in bins]

        bars = ax1.bar(range(len(bins)), recalls, yerr=stds, capsize=5,
                       color=colors, edgecolor='black', alpha=0.8)

        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', fontsize=9, fontweight='bold')

        ax1.set_xticks(range(len(bins)))
        ax1.set_xticklabels(bins)
        ax1.set_ylabel('Mean Parasite Recall', fontsize=11)
        ax1.set_title('Total Parasitemia Stratified', fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.axhline(0.8, color='green', linestyle=':', linewidth=1.5, label='Target: 0.8')
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No total parasitemia data', ha='center', va='center', fontsize=12)

    # Panel 2: Rare Species Stratified
    ax2 = axes[1]
    if rare_stratified:
        recalls = [rare_stratified.get(b, {}).get('mean_recall', 0) for b in bins]
        stds = [rare_stratified.get(b, {}).get('std_recall', 0) for b in bins]
        counts = [rare_stratified.get(b, {}).get('count', 0) for b in bins]

        bars = ax2.bar(range(len(bins)), recalls, yerr=stds, capsize=5,
                       color=['purple'] * 4, edgecolor='black', alpha=0.7)

        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'n={count}', ha='center', fontsize=9, fontweight='bold')

        ax2.set_xticks(range(len(bins)))
        ax2.set_xticklabels(bins)
        ax2.set_ylabel('Mean Rare Species Recall', fontsize=11)
        ax2.set_title('Rare Species at Different Parasitemia\n(P. ovale, P. malariae, P. vivax)', fontsize=12)
        ax2.set_ylim(0, 1.1)
        ax2.axhline(0.8, color='green', linestyle=':', linewidth=1.5, label='Target: 0.8')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No rare species data', ha='center', va='center', fontsize=12)

    plt.tight_layout()
    save_path = save_dir / experiment_name / 'stratified_analysis.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved stratified analysis visualization to: {save_path}")
    return save_path


def plot_per_class_comparison_species(val_results, test_results, config, experiment_name, save_dir):
    """Create per-class comparison visualization for species detection"""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Per-Species Performance: Validation vs Test', fontsize=14, fontweight='bold')

    class_names = list(config.get_class_names().values())
    x = np.arange(len(class_names))
    width = 0.35

    val_pc = val_results.get('per_class', {})
    test_pc = test_results.get('per_class', {})

    # Precision
    val_prec = [val_pc.get(c, {}).get('precision', 0) for c in class_names]
    test_prec = [test_pc.get(c, {}).get('precision', 0) for c in class_names]

    axes[0].bar(x - width/2, val_prec, width, label='Validation', color='skyblue', edgecolor='navy')
    axes[0].bar(x + width/2, test_prec, width, label='Test', color='lightcoral', edgecolor='darkred')
    axes[0].set_ylabel('Precision', fontsize=11)
    axes[0].set_title('Precision', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(axis='y', alpha=0.3)

    # Recall
    val_rec = [val_pc.get(c, {}).get('recall', 0) for c in class_names]
    test_rec = [test_pc.get(c, {}).get('recall', 0) for c in class_names]

    axes[1].bar(x - width/2, val_rec, width, label='Validation', color='skyblue', edgecolor='navy')
    axes[1].bar(x + width/2, test_rec, width, label='Test', color='lightcoral', edgecolor='darkred')
    axes[1].set_ylabel('Recall', fontsize=11)
    axes[1].set_title('Recall', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].legend()
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(axis='y', alpha=0.3)

    # F1-Score
    val_f1 = [val_pc.get(c, {}).get('f1_score', 0) for c in class_names]
    test_f1 = [test_pc.get(c, {}).get('f1_score', 0) for c in class_names]

    axes[2].bar(x - width/2, val_f1, width, label='Validation', color='skyblue', edgecolor='navy')
    axes[2].bar(x + width/2, test_f1, width, label='Test', color='lightcoral', edgecolor='darkred')
    axes[2].set_ylabel('F1-Score', fontsize=11)
    axes[2].set_title('F1-Score', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right')
    axes[2].legend()
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / experiment_name / 'per_class_comparison.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class comparison to: {save_path}")
    return save_path


# ============================================================================
# EVALUATION WITH SPECIES EVALUATOR
# ============================================================================
print("\n" + "=" * 70)
print("RUNNING SPECIES-SPECIFIC EVALUATION")
print("=" * 70)

# Get best model path
best_model_path = run_dir / 'train' / 'weights' / 'best.pt'
if not best_model_path.exists():
    best_model_path = run_dir / 'train' / 'weights' / 'last.pt'

print(f"Evaluating model: {best_model_path}")

# Create species evaluator
eval_output_dir = run_dir / 'evaluation'
eval_output_dir.mkdir(exist_ok=True)

evaluator = SpeciesEvaluator(
    model_path=str(best_model_path),
    dataset_path=str(yolo_path),
    config=config,
    output_dir=str(eval_output_dir),
    yaml_path=str(yaml_path)
)

# Run validation evaluation FIRST (for val-test comparison)
print("\n[VALIDATION SET]")
val_results = evaluator.run_full_evaluation(split='val')

# Run test evaluation
print("\n[TEST SET]")
test_results = evaluator.run_full_evaluation(split='test')

# Print species summary (test results)
evaluator.print_species_summary(test_results)

# Use test_results as primary eval_results for backward compatibility
eval_results = test_results

# ============================================================================
# ADDITIONAL ANALYSIS - Decision Analysis & Recall Variability
# ============================================================================
print("\n" + "=" * 70)
print("RUNNING ADDITIONAL ANALYSIS")
print("=" * 70)

# Get test images for decision analysis
test_img_dir = yolo_path / 'test' / 'images'
test_lbl_dir = yolo_path / 'test' / 'labels'
test_samples = []

for img_path in test_img_dir.glob('*.jpg'):
    label_path = test_lbl_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path) as f:
            if f.read().strip():
                test_samples.append(img_path)

for img_path in test_img_dir.glob('*.png'):
    label_path = test_lbl_dir / f"{img_path.stem}.txt"
    if label_path.exists():
        with open(label_path) as f:
            if f.read().strip():
                test_samples.append(img_path)

print(f"Found {len(test_samples)} test images with annotations")

# Run decision analysis
decision_df, decision_summary = None, None
if test_samples:
    decision_df, decision_summary = analyze_model_decisions_species(
        evaluator.model,
        test_samples,
        f"baseline_{config.model_name}_s{args.seed:03d}",
        run_dir,
        conf_threshold=config.conf,
        iou_threshold=config.iou,
        num_samples_to_display=6
    )

# Compute recall variability
recall_df, recall_var_results = compute_species_recall_variability(
    evaluator.model,
    yolo_path,
    config,
    f"baseline_{config.model_name}_s{args.seed:03d}",
    run_dir
)

# Store recall variability for W&B logging
test_results['recall_variability'] = recall_var_results

# Create additional visualizations
results_base = Path(args.results_dir)
plot_species_stratified_analysis(test_results, f"baseline_{config.model_name}_s{args.seed:03d}", run_dir)
plot_per_class_comparison_species(val_results, test_results, config, f"baseline_{config.model_name}_s{args.seed:03d}", run_dir)

# ============================================================================
# LOG TO WANDB - COMPREHENSIVE LOGGING (Consistent with SA-QGFL and Binary QGFL)
# ============================================================================
if args.use_wandb:
    import pandas as pd

    print("\n" + "="*70)
    print("LOGGING TO WEIGHTS & BIASES")
    print("="*70)

    # Extract metrics for both validation and test
    # Note: evaluator returns 'global' not 'global_metrics'
    val_global = val_results.get('global', {})
    val_per_class = val_results.get('per_class', {})
    val_species = val_results.get('species_analysis', {})

    test_global = test_results.get('global', {})
    test_per_class = test_results.get('per_class', {})
    test_species = test_results.get('species_analysis', {})

    # =====================================
    # SECTION 1: CHARTS (Scalar Metrics)
    # =====================================
    print("\n[1/4] LOGGING CHARTS...")

    # 1.1 Training Metrics (enhanced)
    wandb.log({
        'charts/training/total_time_minutes': training_time / 60,
        'charts/training/total_time_hours': training_time / 3600,
        'charts/training/epochs': args.epochs,
        'charts/training/batch_size': args.batch_size,
    })
    # Add training loss metrics if available
    if hasattr(results, 'results_dict'):
        wandb.log({
            'charts/training/final_train_box_loss': results.results_dict.get('train/box_loss', 0),
            'charts/training/final_train_cls_loss': results.results_dict.get('train/cls_loss', 0),
            'charts/training/final_val_box_loss': results.results_dict.get('val/box_loss', 0),
            'charts/training/final_val_cls_loss': results.results_dict.get('val/cls_loss', 0),
        })

    # 1.2 Global Performance - Validation (NEW)
    wandb.log({
        'charts/validation/mAP50': val_global.get('mAP50', 0),
        'charts/validation/mAP50_95': val_global.get('mAP50_95', 0),
        'charts/validation/precision': val_global.get('precision', 0),
        'charts/validation/recall': val_global.get('recall', 0),
        'charts/validation/f1_score': val_global.get('f1', 0),
    })

    # 1.3 Global Performance - Test
    wandb.log({
        'charts/test/mAP50': test_global.get('mAP50', 0),
        'charts/test/mAP50_95': test_global.get('mAP50_95', 0),
        'charts/test/precision': test_global.get('precision', 0),
        'charts/test/recall': test_global.get('recall', 0),
        'charts/test/f1_score': test_global.get('f1', 0),
    })

    # 1.4 Per-Class Performance Charts - Validation (NEW)
    for class_name, metrics in val_per_class.items():
        prefix = f"charts/validation/{class_name.lower().replace(' ', '_')}"
        wandb.log({
            f'{prefix}/precision': metrics.get('precision', 0),
            f'{prefix}/recall': metrics.get('recall', 0),
            f'{prefix}/f1_score': metrics.get('f1_score', 0),
        })

    # 1.5 Per-Class Performance Charts - Test
    for class_name, metrics in test_per_class.items():
        prefix = f"charts/test/{class_name.lower().replace(' ', '_')}"
        wandb.log({
            f'{prefix}/precision': metrics.get('precision', 0),
            f'{prefix}/recall': metrics.get('recall', 0),
            f'{prefix}/f1_score': metrics.get('f1_score', 0),
            f'{prefix}/support': metrics.get('support', 0),
        })

    # 1.6 Species-Specific Metrics (Rare Species & All Parasites)
    if 'rare_species_metrics' in test_species:
        rare = test_species['rare_species_metrics']
        wandb.log({
            'charts/species/rare_avg_recall': rare.get('avg_recall', 0),
            'charts/species/rare_avg_precision': rare.get('avg_precision', 0),
            'charts/species/rare_avg_f1': rare.get('avg_f1', 0),
        })

    if 'all_parasites' in test_species:
        parasites = test_species['all_parasites']
        wandb.log({
            'charts/species/all_parasites_recall': parasites.get('recall', 0),
            'charts/species/all_parasites_precision': parasites.get('precision', 0),
            'charts/species/all_parasites_f1': parasites.get('f1_score', 0),
        })

    # =====================================
    # SECTION 2: TABLES (Structured Data)
    # =====================================
    print("\n[2/4] CREATING AND LOGGING TABLES...")

    # 2.1 Per-Class Performance Table - Validation (NEW)
    val_class_data = []
    for class_name, metrics in val_per_class.items():
        val_class_data.append({
            'Class': class_name,
            'Precision': float(metrics.get('precision', 0)),
            'Recall': float(metrics.get('recall', 0)),
            'F1': float(metrics.get('f1_score', 0)),
            'mAP50-95': float(val_global.get('per_class_map', {}).get(class_name, 0)),
            'Support': int(metrics.get('support', 0))
        })
    if val_class_data:
        wandb.log({"tables/validation_per_class": wandb.Table(dataframe=pd.DataFrame(val_class_data))})
        print("  ✓ Logged validation_per_class table")

    # 2.2 Per-Class Performance Table - Test
    test_class_data = []
    for class_name, metrics in test_per_class.items():
        test_class_data.append({
            'Class': class_name,
            'Precision': float(metrics.get('precision', 0)),
            'Recall': float(metrics.get('recall', 0)),
            'F1': float(metrics.get('f1_score', 0)),
            'mAP50-95': float(test_global.get('per_class_map', {}).get(class_name, 0)),
            'Support': int(metrics.get('support', 0)),
            'TP': int(metrics.get('tp', 0)),
            'FP': int(metrics.get('fp', 0)),
            'FN': int(metrics.get('fn', 0))
        })
    if test_class_data:
        wandb.log({"tables/test_per_class": wandb.Table(dataframe=pd.DataFrame(test_class_data))})
        print("  ✓ Logged test_per_class table")

    # 2.3 Species Analysis Table (Parasite classes only)
    species_data = []
    for species_name in ['P_falciparum', 'P_ovale', 'P_malariae', 'P_vivax']:
        if species_name in test_species:
            s = test_species[species_name]
            species_data.append({
                'Species': species_name,
                'Recall': float(s.get('recall', 0)),
                'Precision': float(s.get('precision', 0)),
                'F1': float(s.get('f1_score', 0)),
                'mAP50-95': float(s.get('mAP50_95', 0)),
                'Support': int(s.get('support', 0)),
                'Is_Rare': species_name in ['P_ovale', 'P_malariae', 'P_vivax']
            })
    if species_data:
        wandb.log({"tables/species_analysis": wandb.Table(dataframe=pd.DataFrame(species_data))})
        print("  ✓ Logged species_analysis table")

    # 2.4 Error Analysis Table - Aggregate (renamed for consistency)
    if 'errors' in test_results:
        error_data = []
        for error_type, stats in test_results['errors'].items():
            if isinstance(stats, dict):
                error_data.append({
                    'Error_Type': error_type,
                    'Count': stats.get('count', 0),
                    'Rate_Per_Image': stats.get('rate_per_image', 0),
                    'Mean': stats.get('mean', 0),
                    'Std': stats.get('std', 0)
                })
        if error_data:
            wandb.log({"tables/error_analysis_aggregate": wandb.Table(dataframe=pd.DataFrame(error_data))})
            print("  ✓ Logged error_analysis_aggregate table")

    # 2.5 Error Analysis Table - Per Class (NEW)
    if 'errors_per_class' in test_results:
        error_per_class_data = []
        for class_name, class_errors in test_results['errors_per_class'].items():
            if isinstance(class_errors, dict):
                error_per_class_data.append({
                    'Class': class_name,
                    'Classification_Count': class_errors.get('classification', {}).get('count', 0),
                    'Classification_Rate': class_errors.get('classification', {}).get('rate_per_image', 0),
                    'Localization_Count': class_errors.get('localization', {}).get('count', 0),
                    'Background_FP_Count': class_errors.get('background', {}).get('count', 0),
                    'Missed_FN_Count': class_errors.get('missed', {}).get('count', 0),
                    'Duplicate_Count': class_errors.get('duplicate', {}).get('count', 0),
                })
        if error_per_class_data:
            wandb.log({"tables/error_analysis_per_class": wandb.Table(dataframe=pd.DataFrame(error_per_class_data))})
            print("  ✓ Logged error_analysis_per_class table")

    # 2.6 Stratified Analysis Table
    if 'stratified' in test_results:
        strat_data = []
        for bin_name, stats in test_results['stratified'].items():
            if isinstance(stats, dict):
                strat_data.append({
                    'Bin': bin_name,
                    'Mean_Recall': stats.get('mean_recall', 0),
                    'Std_Recall': stats.get('std_recall', 0),
                    'Count': stats.get('count', 0),
                })
        if strat_data:
            wandb.log({"tables/stratified_analysis": wandb.Table(dataframe=pd.DataFrame(strat_data))})
            print("  ✓ Logged stratified_analysis table")

    # 2.6b Rare Species Stratified Table (NEW - WHO-aligned SA-QGFL contribution)
    if 'rare_species_stratified' in test_results:
        rare_strat_data = []
        for bin_name, stats in test_results['rare_species_stratified'].items():
            if isinstance(stats, dict) and not bin_name.startswith('_'):
                rare_strat_data.append({
                    'Parasitemia_Bin': bin_name,
                    'Rare_Species_Recall': stats.get('mean_recall', 0),
                    'Std': stats.get('std_recall', 0),
                    'Median_Recall': stats.get('median_recall', 0),
                    'Image_Count': stats.get('count', 0),
                })
        if rare_strat_data:
            wandb.log({"tables/rare_species_stratified": wandb.Table(dataframe=pd.DataFrame(rare_strat_data))})
            print("  ✓ Logged rare_species_stratified table")

    # 2.7 Recall Variability Table - Per Image (NEW)
    if 'recall_variability' in test_results and 'wandb_table' in test_results['recall_variability']:
        recall_df = pd.DataFrame(test_results['recall_variability']['wandb_table'])
        wandb.log({"tables/recall_variability_per_image": wandb.Table(dataframe=recall_df)})
        print("  ✓ Logged recall_variability_per_image table")

    # 2.8 Confusion Matrix Table (renamed for consistency)
    if 'confusion' in test_results:
        cm = np.array(test_results['confusion'])
        class_names_list = list(config.get_class_names().values())
        if cm.shape[0] > len(class_names_list):
            class_names_list.append('Background/Missed')
        cm_df = pd.DataFrame(cm, index=class_names_list[:cm.shape[0]], columns=class_names_list[:cm.shape[1]])
        wandb.log({"tables/test_confusion_matrix": wandb.Table(dataframe=cm_df)})
        print("  ✓ Logged test_confusion_matrix table")

    # 2.9 Val-Test Comparison Table (NEW) - Species-adapted
    comparison_data = []
    comparison_data.append({
        'Split': 'Validation',
        'mAP50': float(val_global.get('mAP50', 0)),
        'mAP50_95': float(val_global.get('mAP50_95', 0)),
        'Uninfected_F1': float(val_per_class.get('Uninfected', {}).get('f1_score', 0)),
        'P_falciparum_F1': float(val_per_class.get('P_falciparum', {}).get('f1_score', 0)),
        'Rare_Species_Avg_F1': float(val_species.get('rare_species_metrics', {}).get('avg_f1', 0)),
    })
    comparison_data.append({
        'Split': 'Test',
        'mAP50': float(test_global.get('mAP50', 0)),
        'mAP50_95': float(test_global.get('mAP50_95', 0)),
        'Uninfected_F1': float(test_per_class.get('Uninfected', {}).get('f1_score', 0)),
        'P_falciparum_F1': float(test_per_class.get('P_falciparum', {}).get('f1_score', 0)),
        'Rare_Species_Avg_F1': float(test_species.get('rare_species_metrics', {}).get('avg_f1', 0)),
    })
    comparison_data.append({
        'Split': 'Difference (Val-Test)',
        'mAP50': comparison_data[0]['mAP50'] - comparison_data[1]['mAP50'],
        'mAP50_95': comparison_data[0]['mAP50_95'] - comparison_data[1]['mAP50_95'],
        'Uninfected_F1': comparison_data[0]['Uninfected_F1'] - comparison_data[1]['Uninfected_F1'],
        'P_falciparum_F1': comparison_data[0]['P_falciparum_F1'] - comparison_data[1]['P_falciparum_F1'],
        'Rare_Species_Avg_F1': comparison_data[0]['Rare_Species_Avg_F1'] - comparison_data[1]['Rare_Species_Avg_F1'],
    })
    wandb.log({"tables/val_test_comparison": wandb.Table(dataframe=pd.DataFrame(comparison_data))})
    print("  ✓ Logged val_test_comparison table")

    # 2.10 Precision-Recall Analysis Table (NEW)
    if 'pr_analysis' in test_results:
        pr_data = []
        for class_name, pr_stats in test_results['pr_analysis'].items():
            pr_data.append({
                'Class': class_name,
                'AP': pr_stats.get('ap', 0),
                'mAP50-95': float(test_global.get('per_class_map', {}).get(class_name, 0)),
                'Optimal_Threshold': pr_stats.get('optimal_threshold', 0),
                'Precision_at_Optimal': pr_stats.get('precision_at_optimal', 0),
                'Recall_at_Optimal': pr_stats.get('recall_at_optimal', 0),
                'Max_F1': pr_stats.get('max_f1', 0)
            })
        if pr_data:
            wandb.log({"tables/precision_recall_analysis": wandb.Table(dataframe=pd.DataFrame(pr_data))})
            print("  ✓ Logged precision_recall_analysis table")

    # 2.11 Decision Analysis Table (NEW)
    decision_csv_table = run_dir / f"baseline_{config.model_name}_s{args.seed:03d}" / 'decision_analysis' / 'decision_analysis_complete.csv'
    if decision_csv_table.exists():
        try:
            decision_df_table = pd.read_csv(decision_csv_table)
            wandb.log({"tables/decision_analysis": wandb.Table(dataframe=decision_df_table)})
            print("  ✓ Logged decision_analysis table")
        except Exception as e:
            print(f"  Warning: Could not log decision_analysis table: {e}")

    # 2.12 Rare Species Summary Table
    rare_summary = []
    if 'rare_species_metrics' in test_species:
        rare = test_species['rare_species_metrics']
        rare_summary.append({
            'Metric': 'Rare Species Average',
            'Recall': rare.get('avg_recall', 0),
            'Precision': rare.get('avg_precision', 0),
            'F1': rare.get('avg_f1', 0),
            'Species': ', '.join(rare.get('species_included', []))
        })
    if 'all_parasites' in test_species:
        parasites = test_species['all_parasites']
        rare_summary.append({
            'Metric': 'All Parasites Combined',
            'Recall': parasites.get('recall', 0),
            'Precision': parasites.get('precision', 0),
            'F1': parasites.get('f1_score', 0),
            'Species': 'P_falciparum, P_ovale, P_malariae, P_vivax'
        })
    if rare_summary:
        wandb.log({"tables/rare_species_summary": wandb.Table(dataframe=pd.DataFrame(rare_summary))})
        print("  ✓ Logged rare_species_summary table")

    # =====================================
    # SECTION 3: ARTIFACTS & IMAGES
    # =====================================
    print("\n[3/4] LOGGING ARTIFACTS AND VISUALIZATIONS...")

    # 3.1 Create main artifact for all visualizations
    experiment_name = f"baseline_{config.model_name}_s{args.seed:03d}"
    viz_artifact = wandb.Artifact(
        name=f"visualizations_{experiment_name}",
        type="visualizations",
        metadata={
            'model': config.model_name,
            'task': config.task,
            'loss_type': 'default',
            'seed': args.seed
        }
    )

    # 3.2 Add training images from run directory
    viz_count = 0
    train_dir = run_dir / 'train'
    if train_dir.exists():
        for img_file in train_dir.glob('*.png'):
            viz_artifact.add_file(str(img_file))
            wandb.log({f"images/training/{img_file.stem}": wandb.Image(str(img_file))})
            viz_count += 1
        for img_file in train_dir.glob('*.jpg'):
            viz_artifact.add_file(str(img_file))
            viz_count += 1

    # 3.3 Add evaluation images
    if eval_output_dir.exists():
        for img_file in Path(eval_output_dir).glob('*.png'):
            viz_artifact.add_file(str(img_file))
            wandb.log({f"images/evaluation/{img_file.stem}": wandb.Image(str(img_file))})
            viz_count += 1

    # 3.4 Add decision analysis images (NEW)
    decision_path = run_dir / f"baseline_{config.model_name}_s{args.seed:03d}" / 'decision_analysis'
    if decision_path.exists():
        for idx, img_file in enumerate(sorted(decision_path.glob('decision_*.png'))):
            viz_artifact.add_file(str(img_file))
            if idx < 4:  # Only log first 4 to W&B for quick viewing
                wandb.log({f"images/decision/{img_file.stem}": wandb.Image(str(img_file))})
            viz_count += 1

    # 3.5 Add custom analysis visualizations
    experiment_subdir = run_dir / f"baseline_{config.model_name}_s{args.seed:03d}"
    custom_viz_files = [
        ('stratified_analysis.png', 'images/stratified_analysis'),
        ('per_class_comparison.png', 'images/per_class_comparison'),
        ('recall_variability.png', 'images/recall_variability'),
    ]
    for filename, wandb_key in custom_viz_files:
        viz_file = experiment_subdir / filename
        if viz_file.exists():
            viz_artifact.add_file(str(viz_file))
            wandb.log({wandb_key: wandb.Image(str(viz_file))})
            viz_count += 1
            print(f"  ✓ Logged {filename}")

    if viz_count > 0:
        wandb.log_artifact(viz_artifact)
        print(f"  ✓ Logged visualization artifact with {viz_count} images")

    # 3.5 Model artifact (enhanced metadata)
    if best_model_path.exists():
        model_artifact = wandb.Artifact(
            name=f"{config.dataset}_{config.task}_{config.model_name}_baseline_model",
            type="model",
            metadata={
                # Performance metrics
                'test_mAP50': test_global.get('mAP50', 0),
                'test_mAP50_95': test_global.get('mAP50_95', 0),
                # Species-specific
                'p_falciparum_recall': test_per_class.get('P_falciparum', {}).get('recall', 0),
                'p_falciparum_f1': test_per_class.get('P_falciparum', {}).get('f1_score', 0),
                'rare_species_avg_recall': test_species.get('rare_species_metrics', {}).get('avg_recall', 0),
                'rare_species_avg_f1': test_species.get('rare_species_metrics', {}).get('avg_f1', 0),
                # Training info
                'loss_type': 'default',
                'seed': args.seed,
                'epochs_trained': args.epochs,
                'training_time_minutes': training_time / 60,
            }
        )
        model_artifact.add_file(str(best_model_path))
        wandb.log_artifact(model_artifact)
        print("  ✓ Logged model artifact")

    # =====================================
    # SECTION 4: DATA ARTIFACTS (CSVs) - NEW
    # =====================================
    print("\n[4/4] LOGGING DATA ARTIFACTS...")

    data_artifact = wandb.Artifact(
        name=f"analysis_data_{experiment_name}",
        type="analysis_data",
        metadata={'model': experiment_name, 'loss_type': 'default'}
    )

    # Add recall variability CSV (updated path)
    experiment_subdir = run_dir / f"baseline_{config.model_name}_s{args.seed:03d}"
    recall_csv = experiment_subdir / 'recall_variability_data.csv'
    if recall_csv.exists():
        data_artifact.add_file(str(recall_csv))
        print("  ✓ Added recall_variability_data.csv")

    # Add decision analysis CSV (updated path)
    decision_csv = experiment_subdir / 'decision_analysis' / 'decision_analysis_complete.csv'
    if decision_csv.exists():
        data_artifact.add_file(str(decision_csv))
        print("  ✓ Added decision_analysis_complete.csv")

    # Add evaluation report
    eval_report = eval_output_dir / 'evaluation_report.txt'
    if eval_report.exists():
        data_artifact.add_file(str(eval_report))
        print("  ✓ Added evaluation_report.txt")

    wandb.log_artifact(data_artifact)

    # Log training time
    wandb.log({'training_time_hours': training_time / 3600})

    print("\n" + "="*70)
    print("W&B LOGGING COMPLETE")
    print("="*70)
    print(f"✓ Charts: Training, validation, test, and species-specific metrics")
    print(f"✓ Tables: 12 comprehensive tables created")
    print(f"✓ Artifacts: {viz_count} visualizations + model + data files")

    wandb.finish()

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================
final_results = {
    'experiment': {
        'model': config.model_name,
        'dataset': config.dataset,
        'task': config.task,
        'seed': args.seed,
        'loss_type': 'default',  # BCE for YOLO, VFL for RT-DETR
    },
    'training': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'training_time_hours': training_time / 3600,
    },
    'evaluation': eval_results,
}

results_file = run_dir / 'baseline_results.json'
with open(results_file, 'w') as f:
    json.dump(final_results, f, indent=2, default=str)

print(f"\nResults saved to: {results_file}")

print("\n" + "=" * 70)
print("SPECIES BASELINE TRAINING COMPLETE")
print("=" * 70)
print(f"Model: {config.model_name}")
print(f"Best weights: {best_model_path}")
print(f"Evaluation results: {eval_output_dir}")
print("=" * 70)
