#!/usr/bin/env python3
"""
Modular Training Script for Malaria Detection Experiments
Cluster-ready with collision-free directory management

Usage:
    python train_baseline.py --model yolov8s --dataset d1 --epochs 200
    python train_baseline.py --model yolov11s --dataset d2 --task binary --strategy no_weights
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import time
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import torch
import yaml
import wandb
from ultralytics import YOLO

from configs.baseline_config import ExperimentConfig
from src.utils.paths import get_dataset_paths, verify_dataset
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.training.strategy_wrapper import create_training_strategy
import numpy as np


def run_prevalence_stratified_analysis(
    test_results: dict,
    class_names: dict,
    task: str,
    save_dir: Path,
    use_wandb: bool = False
) -> dict:
    """
    Analyze model performance across different parasitemia (infection density) levels.

    This is THE KEY METRIC from the QGFL paper:
    "QGFL achieves remarkable improvement in detecting infected cells in the
    clinically vital 1–3% parasitaemia range"

    Args:
        test_results: Dictionary from evaluator.evaluate_model('test')
        class_names: Dictionary of class IDs to names
        task: Task type ('binary', 'species', 'staging')
        save_dir: Directory to save visualization
        use_wandb: Whether to log to W&B

    Returns:
        Dictionary with stratified metrics for W&B logging
    """
    # Only run for binary classification task
    if task != 'binary':
        print(f"\nSkipping prevalence-stratified analysis (only for binary task, current: {task})")
        return {}

    print("\n" + "="*70)
    print("PREVALENCE-STRATIFIED ANALYSIS")
    print("="*70)
    print("Analyzing performance across parasitemia levels...")
    print("(This is the key metric for malaria detection quality)")

    # Get stratified results from evaluator
    if 'stratified' not in test_results:
        print("⚠️  No stratified results found in test_results")
        print("   Make sure ComprehensiveEvaluator calculated prevalence bins")
        return {}

    stratified = test_results['stratified']

    # Standard parasitemia bins
    bins = ['0-1%', '1-3%', '3-5%', '>5%']

    # Verify all bins exist
    missing_bins = [b for b in bins if b not in stratified]
    if missing_bins:
        print(f"⚠️  Missing bins: {missing_bins}")
        available_bins = [b for b in bins if b in stratified]
        if not available_bins:
            print("   No stratified data available")
            return {}
        bins = available_bins

    # Display results table
    from tabulate import tabulate

    strat_data = []
    for bin_name in bins:
        stats = stratified[bin_name]
        strat_data.append([
            bin_name,
            f"{stats['mean_recall']:.3f}",
            f"{stats['std_recall']:.3f}",
            stats['count']
        ])

    print("\n" + tabulate(
        strat_data,
        headers=['Parasitemia Level', 'Mean Recall', 'Std Dev', 'N Images'],
        tablefmt='fancy_grid',
        numalign='right'
    ))

    # Clinical significance notes
    print("\nClinical Significance:")
    for bin_name in bins:
        recall = stratified[bin_name]['mean_recall']
        std = stratified[bin_name]['std_recall']
        count = stratified[bin_name]['count']

        clinical_note = ""
        if bin_name == '0-1%':
            clinical_note = " ← Ultra-low (hardest to detect, most critical)"
        elif bin_name == '1-3%':
            clinical_note = " ← CRITICAL RANGE (early detection, key metric)"
        elif bin_name == '3-5%':
            clinical_note = " ← Moderate (routine detection)"
        else:  # >5%
            clinical_note = " ← High (easier detection)"

        print(f"  {bin_name}: {recall:.3f} ± {std:.3f} (n={count}){clinical_note}")

    # Create visualization
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Across Parasitemia Levels', fontsize=14, fontweight='bold')

    recalls = [stratified[b]['mean_recall'] for b in bins]
    stds = [stratified[b]['std_recall'] for b in bins]
    counts = [stratified[b]['count'] for b in bins]

    # Color gradient: red (critical) → green (less critical)
    colors = ['#B71C1C', '#FF6F00', '#FDD835', '#43A047']

    # Subplot 1: Bar chart with error bars
    bars = []
    for i, (bin_name, recall, std, count) in enumerate(zip(bins, recalls, stds, counts)):
        bar = ax1.bar(
            i, recall,
            yerr=std if count > 0 else 0,
            capsize=5,
            color=colors[i] if i < len(colors) else '#666666',
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        bars.append(bar)

    ax1.set_xticks(range(len(bins)))
    ax1.set_xticklabels(bins)
    ax1.set_xlabel('Parasitemia Level (%)', fontsize=12)
    ax1.set_ylabel('Mean Infected Cell Recall', fontsize=12)
    ax1.set_title('Recall by Parasitemia Level', fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add clinical threshold line
    ax1.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.text(len(bins)-0.5, 0.82, 'Target: 0.8', fontsize=9, color='green')

    # Add sample counts on bars
    for i, (bar_container, count) in enumerate(zip(bars, counts)):
        if bar_container:
            bar = bar_container[0]
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2, height + 0.02,
                f'n={count}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold'
            )

    # Subplot 2: Line plot with confidence intervals
    x_pos = np.arange(len(bins))

    ax2.plot(
        x_pos, recalls,
        'o-',
        markersize=10,
        linewidth=2.5,
        color='#D32F2F',
        markeredgecolor='black',
        markeredgewidth=1,
        label='Mean Recall'
    )

    # Error bars
    ax2.errorbar(
        x_pos, recalls,
        yerr=stds,
        fmt='none',
        ecolor='#D32F2F',
        alpha=0.3,
        capsize=5,
        capthick=2
    )

    # Confidence interval shading
    ax2.fill_between(
        x_pos,
        [max(0, r-s) for r, s in zip(recalls, stds)],
        [min(1, r+s) for r, s in zip(recalls, stds)],
        alpha=0.15,
        color='#D32F2F'
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bins)
    ax2.set_xlabel('Parasitemia Level (%)', fontsize=12)
    ax2.set_ylabel('Mean Infected Cell Recall', fontsize=12)
    ax2.set_title('Performance Trend Across Parasitemia Levels', fontsize=12)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right')

    # Add clinical threshold line
    ax2.axhline(y=0.8, color='green', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.tight_layout()

    # Save figure
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / 'prevalence_stratified_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")
    plt.close(fig)

    # Prepare return data for W&B
    stratified_summary = {
        'bins': {},
        'clinical_assessment': {},
        'figure_path': str(save_path)
    }

    # Per-bin metrics
    for bin_name in bins:
        stratified_summary['bins'][bin_name] = {
            'mean_recall': float(stratified[bin_name]['mean_recall']),
            'std_recall': float(stratified[bin_name]['std_recall']),
            'count': int(stratified[bin_name]['count'])
        }

    # Clinical assessments
    if '1-3%' in stratified and stratified['1-3%']['count'] > 0:
        critical_recall = stratified['1-3%']['mean_recall']
        stratified_summary['clinical_assessment']['critical_range_recall'] = float(critical_recall)
        stratified_summary['clinical_assessment']['meets_target'] = critical_recall >= 0.8

        if critical_recall < 0.5:
            status = "POOR - Fails to detect early infections"
        elif critical_recall < 0.7:
            status = "FAIR - Misses many early infections"
        elif critical_recall < 0.8:
            status = "GOOD - Close to clinical target"
        else:
            status = "EXCELLENT - Meets clinical requirements"

        stratified_summary['clinical_assessment']['status'] = status
        print(f"\nClinical Assessment (1-3% range): {status}")

    print("\n" + "="*70)

    # Log to W&B if requested
    if use_wandb:
        try:
            # Log metrics
            for bin_name in bins:
                wandb.log({
                    f'stratified/{bin_name.replace("%", "pct")}/recall': stratified[bin_name]['mean_recall'],
                    f'stratified/{bin_name.replace("%", "pct")}/std': stratified[bin_name]['std_recall'],
                    f'stratified/{bin_name.replace("%", "pct")}/count': stratified[bin_name]['count'],
                })

            # Log figure
            wandb.log({'stratified/performance_plot': wandb.Image(str(save_path))})
            print("✓ Logged to W&B")
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")

    return stratified_summary


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train YOLO models for malaria detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       choices=['yolov8s', 'yolov11s'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['d1', 'd2', 'd3'],
                       help='Dataset to use')

    # Optional arguments
    parser.add_argument('--task', type=str, default='binary',
                       choices=['binary', 'species', 'staging'],
                       help='Classification task')
    parser.add_argument('--strategy', type=str, default='no_weights',
                       choices=['no_weights', 'qgfl'],
                       help='Training strategy')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')

    # Paths
    parser.add_argument('--results-dir', type=str, default='../results',
                       help='Base directory for results')
    parser.add_argument('--runs-dir', type=str, default='../runs/detect',
                       help='Base directory for YOLO runs')

    # W&B
    parser.add_argument('--use-wandb', action='store_true', default=True,
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='malaria_qgfl_experiments',
                       help='W&B project name')
    parser.add_argument('--wandb-key', type=str,
                       default='4024481dae024d54316dd81438ccca923991c6ec',
                       help='W&B API key')

    # Evaluation
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation step (training only)')

    # System
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu/mps). Default: auto-detect')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Experiment naming
    parser.add_argument('--name-suffix', type=str, default='',
                       help='Optional suffix for experiment name')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Disable timestamp in experiment name')

    return parser.parse_args()


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"✓ Random seed set to {seed}")


def create_experiment_name(args):
    """Create unique experiment name with timestamp"""
    base_name = f"{args.model}_{args.dataset}_{args.task}_{args.strategy}"

    if args.name_suffix:
        base_name += f"_{args.name_suffix}"

    if not args.no_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name += f"_{timestamp}"

    return base_name


def get_class_distribution(yolo_path, class_names):
    """Calculate class distribution from training labels"""
    distribution = defaultdict(int)
    label_dir = yolo_path / 'train' / 'labels'

    for class_id, class_name in class_names.items():
        distribution[class_name] = 0

    for label_file in label_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_name = class_names[class_id]
                    distribution[class_name] += 1

    return dict(distribution)


def setup_directories(args, experiment_name):
    """Create collision-free directory structure"""
    # Create base directories
    results_base = Path(args.results_dir)
    runs_base = Path(args.runs_dir)

    results_base.mkdir(parents=True, exist_ok=True)
    runs_base.mkdir(parents=True, exist_ok=True)

    # Create experiment-specific directories
    run_dir = runs_base / experiment_name
    results_dir = results_base / experiment_name

    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, results_dir


def prepare_yaml_config(args, yolo_path, class_names):
    """Create or verify YAML configuration file"""
    yaml_dir = Path('configs/data_yamls')
    yaml_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = yaml_dir / f'{args.dataset}_{args.task}.yaml'

    if not yaml_path.exists():
        print(f"Creating YAML config: {yaml_path}")
        data_yaml = {
            'path': str(yolo_path),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': class_names,
            'nc': len(class_names)
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    else:
        print(f"✓ Using existing YAML: {yaml_path}")

    return yaml_path


def train(args):
    """Main training function"""
    print("="*70)
    print("MALARIA DETECTION TRAINING PIPELINE")
    print("="*70)

    # Set random seed
    set_seed(args.seed)

    # Create experiment name
    experiment_name = create_experiment_name(args)
    print(f"\nExperiment: {experiment_name}")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task: {args.task}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Epochs: {args.epochs}")

    # Create ExperimentConfig for compatibility
    config = ExperimentConfig(
        dataset=args.dataset,
        task=args.task,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_key=args.wandb_key
    )

    # Setup directories (collision-free)
    run_dir, results_dir = setup_directories(args, experiment_name)
    print(f"\nDirectories:")
    print(f"  Run dir: {run_dir}")
    print(f"  Results dir: {results_dir}")

    # Verify and prepare dataset
    print(f"\n{'='*70}")
    print("DATASET PREPARATION")
    print("="*70)

    dataset_valid = verify_dataset(args.dataset, args.task)
    if not dataset_valid:
        raise RuntimeError(f"Dataset preparation failed for {args.dataset}/{args.task}")

    yolo_path = get_dataset_paths(args.dataset, args.task)
    class_names = config.get_class_names()

    # Create YAML config
    yaml_path = prepare_yaml_config(args, yolo_path, class_names)

    # Calculate class distribution
    print(f"\n{'='*70}")
    print("CLASS DISTRIBUTION ANALYSIS")
    print("="*70)

    class_distribution = get_class_distribution(yolo_path, class_names)
    total_annotations = sum(class_distribution.values())

    print(f"\nTraining set annotations: {total_annotations}")
    for class_name, count in class_distribution.items():
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

    # Create training strategy
    print(f"\n{'='*70}")
    print(f"TRAINING STRATEGY: {args.strategy.upper()}")
    print("="*70)

    strategy = create_training_strategy(args.strategy, config, class_distribution)
    strategy_params = strategy.get_training_params()
    hyperparameter_adjustments = strategy.get_hyperparameter_adjustments()

    print(f"Class weights: {strategy.class_weights.numpy()}")
    print(f"Minority classes: {[class_names[i] for i in strategy.minority_classes]}")
    print(f"Loss weights - Box: {strategy_params['box']}, Cls: {strategy_params['cls']}, DFL: {strategy_params['dfl']}")

    # Initialize W&B
    wandb_run = None
    if args.use_wandb:
        print(f"\n{'='*70}")
        print("WEIGHTS & BIASES INITIALIZATION")
        print("="*70)

        wandb.login(key=args.wandb_key)

        # Generate unique W&B run ID for resumability
        wandb_id = wandb.util.generate_id()

        wandb_config = {
            # Model
            'model': args.model,
            'training_strategy': args.strategy,

            # Training
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'imgsz': args.imgsz,
            'learning_rate': config.lr0,
            'optimizer': config.optimizer,
            'momentum': config.momentum,
            'weight_decay': config.weight_decay,

            # Dataset
            'dataset': args.dataset,
            'task': args.task,
            'num_classes': len(class_names),
            'class_distribution': class_distribution,
            'total_annotations': total_annotations,

            # System
            'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
            'workers': args.workers,
            'seed': args.seed,

            # Experiment
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'run_dir': str(run_dir),
        }

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            id=wandb_id,
            config=wandb_config,
            tags=[args.dataset, args.task, args.model, args.strategy],
            resume='allow'
        )

        # Save W&B ID for reference
        with open(run_dir / 'wandb_id.txt', 'w') as f:
            f.write(wandb_id)

        print(f"✓ W&B initialized: {wandb_run.name}")
        print(f"✓ Run URL: {wandb_run.url}")
        print(f"✓ Run ID saved to: {run_dir / 'wandb_id.txt'}")

    # Load model
    print(f"\n{'='*70}")
    print("MODEL INITIALIZATION")
    print("="*70)

    model_path = f'{args.model}.pt'
    print(f"Loading pretrained weights: {model_path}")
    model = YOLO(model_path)

    # Log model info to W&B
    if args.use_wandb:
        try:
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            wandb.config.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)
            })
            print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        except:
            pass

    # Apply hyperparameter adjustments from strategy
    for param, value in hyperparameter_adjustments.items():
        if hasattr(config, param):
            old_value = getattr(config, param)
            setattr(config, param, value)
            print(f"Adjusted {param}: {old_value} → {value}")

    # Prepare training arguments
    train_args = {
        'data': str(yaml_path),
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch_size,
        'name': experiment_name,
        'project': args.runs_dir,  # Controls base save location
        'device': args.device or (0 if torch.cuda.is_available() else 'cpu'),
        'patience': getattr(config, 'patience', 20),
        'save': True,
        'save_period': getattr(config, 'save_period', 10),
        'val': True,
        'amp': True,
        'exist_ok': True,  # Allow continuation if interrupted
        'seed': args.seed,
        'deterministic': True,
        'workers': args.workers,

        # Optimizer
        'optimizer': config.optimizer,
        'lr0': config.lr0,
        'lrf': 0.01,
        'momentum': config.momentum,
        'weight_decay': config.weight_decay,
        'warmup_epochs': getattr(config, 'warmup_epochs', 3.0),
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,

        # Loss weights from strategy
        'box': strategy_params.get('box', 7.5),
        'cls': strategy_params.get('cls', 0.5),
        'dfl': strategy_params.get('dfl', 1.5),

        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,

        # Other
        'plots': True,
        'verbose': True,
    }

    # Save training configuration
    config_save_path = run_dir / 'train_config.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'train_args': {k: str(v) if isinstance(v, Path) else v for k, v in train_args.items()},
            'class_distribution': class_distribution,
            'strategy_params': strategy_params,
        }, f, indent=2)
    print(f"\n✓ Training config saved to: {config_save_path}")

    # Start training
    print(f"\n{'='*70}")
    print(f"TRAINING ({args.epochs} EPOCHS)")
    print("="*70)
    print(f"Monitor output for real-time progress...")
    print(f"Results will be saved to: {run_dir}")
    print("="*70 + "\n")

    training_start_time = time.time()

    try:
        results = model.train(**train_args)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.use_wandb:
            wandb.alert(title="Training Failed", text=str(e))
        raise

    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Total time: {total_training_time/60:.2f} minutes")
    print(f"Average per epoch: {total_training_time/args.epochs:.2f} seconds")

    # Get model paths
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    last_model_path = Path(results.save_dir) / 'weights' / 'last.pt'
    results_csv = Path(results.save_dir) / 'results.csv'

    print(f"\nModel weights:")
    print(f"  Best: {best_model_path}")
    print(f"  Last: {last_model_path}")
    print(f"  Results CSV: {results_csv}")

    # Log training curves to W&B
    if args.use_wandb and results_csv.exists():
        print(f"\n{'='*70}")
        print("LOGGING TO WEIGHTS & BIASES")
        print("="*70)

        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = [col.strip() for col in df.columns]

        print(f"Logging {len(df)} epochs of metrics...")

        for idx, row in df.iterrows():
            epoch = int(row['epoch'])

            metrics = {
                'epoch': epoch,
                'train/box_loss': float(row['train/box_loss']) if 'train/box_loss' in row else None,
                'train/cls_loss': float(row['train/cls_loss']) if 'train/cls_loss' in row else None,
                'train/dfl_loss': float(row['train/dfl_loss']) if 'train/dfl_loss' in row else None,
                'val/box_loss': float(row['val/box_loss']) if 'val/box_loss' in row else None,
                'val/cls_loss': float(row['val/cls_loss']) if 'val/cls_loss' in row else None,
                'val/dfl_loss': float(row['val/dfl_loss']) if 'val/dfl_loss' in row else None,
                'metrics/precision': float(row['metrics/precision(B)']) if 'metrics/precision(B)' in row else None,
                'metrics/recall': float(row['metrics/recall(B)']) if 'metrics/recall(B)' in row else None,
                'metrics/mAP50': float(row['metrics/mAP50(B)']) if 'metrics/mAP50(B)' in row else None,
                'metrics/mAP50-95': float(row['metrics/mAP50-95(B)']) if 'metrics/mAP50-95(B)' in row else None,
            }

            # Remove None values
            metrics = {k: v for k, v in metrics.items() if v is not None}
            wandb.log(metrics)

        print("✓ Training curves logged to W&B")

    # Evaluation (if not skipped)
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print("EVALUATION")
        print("="*70)

        evaluator = ComprehensiveEvaluator(
            model_path=str(best_model_path),
            data_yaml_path=str(yaml_path),
            class_names=class_names,
            conf_threshold=0.001,
            iou_threshold=0.5
        )

        print("\nRunning comprehensive evaluation on test set...")
        eval_results = evaluator.evaluate_model('test')

        # Save evaluation results
        eval_save_path = results_dir / 'evaluation_results.json'
        with open(eval_save_path, 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        print(f"✓ Evaluation results saved to: {eval_save_path}")

        # Log to W&B
        if args.use_wandb:
            wandb.log({
                'test/precision': eval_results['global_metrics']['precision'],
                'test/recall': eval_results['global_metrics']['recall'],
                'test/mAP50': eval_results['global_metrics']['map50'],
                'test/mAP50-95': eval_results['global_metrics']['map50_95'],
            })
            print("✓ Evaluation metrics logged to W&B")

        # Run prevalence-stratified analysis (Cell 14)
        stratified_summary = run_prevalence_stratified_analysis(
            test_results=eval_results,
            class_names=class_names,
            task=args.task,
            save_dir=results_dir,
            use_wandb=args.use_wandb
        )

    # Finish W&B run
    if args.use_wandb:
        wandb.log({
            'training/total_time_minutes': total_training_time / 60,
            'training/time_per_epoch_seconds': total_training_time / args.epochs,
        })
        wandb.finish()
        print("\n✓ W&B run finished")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to: {run_dir}")
    print(f"Evaluation results: {results_dir}")

    return {
        'experiment_name': experiment_name,
        'run_dir': run_dir,
        'results_dir': results_dir,
        'best_model': best_model_path,
        'training_time': total_training_time,
    }


def main():
    """Main entry point"""
    args = parse_args()

    # Print configuration
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*70 + "\n")

    # Run training
    results = train(args)

    print(f"\n✓ Experiment '{results['experiment_name']}' completed successfully")
    print(f"✓ Best model: {results['best_model']}")
    print(f"✓ Total time: {results['training_time']/60:.2f} minutes\n")


if __name__ == '__main__':
    main()
