# src/evaluation/evaluator.py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from tqdm import tqdm
import json
from PIL import Image
from ultralytics import YOLO
from tabulate import tabulate
import torch

class ComprehensiveEvaluator:
    """Complete evaluation suite matching all paper requirements"""
    
    def __init__(self, model_path, dataset_path, config, output_dir="evaluation_results"):
        self.model = YOLO(model_path) if isinstance(model_path, (str, Path)) else model_path
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = config.get_class_names()
        self.num_classes = len(self.class_names)
        
        # For visualizations
        self.colors = self._generate_colors()
        
        # Store per-class errors
        self.errors_per_class = {}
        
    def _generate_colors(self):
        """Generate distinct colors for each class"""
        if self.num_classes == 2:
            return [[0, 1, 0, 1], [1, 0, 0, 1]]  # Green, Red
        else:
            cmap = plt.cm.get_cmap('tab10' if self.num_classes <= 10 else 'tab20')
            return [cmap(i % cmap.N)[:4] for i in range(self.num_classes)]
    
    def run_full_evaluation(self, split='test'):
        """Run complete evaluation pipeline"""
        results = {
            'split': split,
            'dataset': self.config.dataset,
            'task': self.config.task,
            'model': self.config.model_name
        }
        
        print("\n" + "="*70)
        print(f"COMPREHENSIVE EVALUATION - {split.upper()} SET")
        print(f"Dataset: {self.config.dataset.upper()} | Task: {self.config.task}")
        print("="*70)
        
        # 1. Global metrics
        print("\n[1/6] Computing global metrics...")
        results['global'] = self.compute_global_metrics(split)
        
        # 2. Per-class metrics with matching
        print("\n[2/6] Computing per-class metrics...")
        results['per_class'] = self.compute_per_class_metrics(split)
        
        # 3. Precision-Recall curves (UPDATED METHOD)
        print("\n[3/6] Computing precision-recall curves...")
        results['pr_analysis'] = self.compute_pr_curves(split)
        
        # 4. Prevalence-stratified analysis (Critical from paper)
        print("\n[4/6] Running prevalence-stratified analysis...")
        results['stratified'] = self.compute_stratified_analysis(split)
        
        # 5. TIDE Error analysis
        print("\n[5/6] Running TIDE error analysis...")
        results['errors'] = self.compute_error_analysis(split)
        results['errors_per_class'] = self.errors_per_class  # Store per-class errors
        
        # 6. Confusion matrix
        print("\n[6/6] Computing confusion matrix...")
        results['confusion'] = self.compute_confusion_matrix(split)
        
        # Save results
        self.save_results(results)
        
        return results
    
    def compute_global_metrics(self, split):
        """Compute overall metrics using YOLO validation"""
        yaml_path = Path(f'../configs/data_yamls/{self.config.dataset}_{self.config.task}.yaml')
        
        metrics = self.model.val(
            data=str(yaml_path),
            split=split,
            conf=0.5,
            iou=0.5,
            verbose=False
        )
        
        return {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0,
            'recall': float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0,
        }
    
    def compute_per_class_metrics(self, split):
        """Detailed per-class analysis with proper matching"""
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        # Initialize counters
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for img_path in tqdm(img_files, desc="Analyzing per-class metrics"):
            # Load ground truth
            label_path = lbl_dir / (img_path.stem + '.txt')
            gt_boxes = []
            
            if label_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            x1 = x_center - width/2
                            y1 = y_center - height/2
                            x2 = x_center + width/2
                            y2 = y_center + height/2
                            
                            gt_boxes.append({
                                'class_id': class_id,
                                'box': [x1, y1, x2, y2]
                            })
            
            # Get predictions
            results = self.model.predict(img_path, conf=0.5, iou=0.5, verbose=False)[0]
            pred_boxes = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pred_boxes.append({
                        'class_id': class_id,
                        'box': [x1, y1, x2, y2],
                        'conf': box.conf.item()
                    })
            
            # Match predictions to ground truth
            matched_gt = [False] * len(gt_boxes)
            matched_pred = [False] * len(pred_boxes)
            
            # For each prediction, find best matching GT
            for pred_idx, pred in enumerate(pred_boxes):
                best_iou = 0.5
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if matched_gt[gt_idx] or gt['class_id'] != pred['class_id']:
                        continue
                    
                    iou = self._compute_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    class_stats[pred['class_id']]['tp'] += 1
                    matched_gt[best_gt_idx] = True
                    matched_pred[pred_idx] = True
                else:
                    class_stats[pred['class_id']]['fp'] += 1
            
            # Count false negatives
            for gt_idx, gt in enumerate(gt_boxes):
                if not matched_gt[gt_idx]:
                    class_stats[gt['class_id']]['fn'] += 1
        
        # Calculate metrics
        metrics = {}
        for class_id in range(self.num_classes):
            tp = class_stats[class_id]['tp']
            fp = class_stats[class_id]['fp']
            fn = class_stats[class_id]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[self.class_names[class_id]] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        return metrics
    
    def compute_pr_curves(self, split):
        """Compute precision-recall curves with FULL curve data for plotting"""
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        # Collect all predictions with scores
        class_predictions = defaultdict(list)
        total_gt = defaultdict(int)
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for img_path in tqdm(img_files, desc="Collecting PR data"):
            # Load GT
            label_path = lbl_dir / (img_path.stem + '.txt')
            gt_boxes = []
            
            if label_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            total_gt[class_id] += 1
                            
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            gt_boxes.append({
                                'class_id': class_id,
                                'box': self._xywh_to_xyxy([x_center, y_center, width, height])
                            })
            
            # Get predictions at low threshold
            results = self.model.predict(img_path, conf=0.01, iou=0.5, verbose=False)[0]
            
            if results.boxes is not None:
                # Match and record
                matched_gt = [False] * len(gt_boxes)
                
                for box in results.boxes:
                    class_id = int(box.cls.item())
                    conf = box.conf.item()
                    pred_box = box.xyxy[0].tolist()
                    
                    # Find best matching GT
                    best_iou = 0.5
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_boxes):
                        if matched_gt[gt_idx] or gt['class_id'] != class_id:
                            continue
                        
                        iou = self._compute_iou(pred_box, gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        class_predictions[class_id].append((conf, 1))  # TP
                        matched_gt[best_gt_idx] = True
                    else:
                        class_predictions[class_id].append((conf, 0))  # FP
        
        # Compute PR curves with FULL data
        pr_results = {}
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            preds = sorted(class_predictions[class_id], key=lambda x: x[0], reverse=True)
            
            if not preds:
                pr_results[class_name] = {
                    'ap': 0, 
                    'optimal_threshold': 0.5,
                    'max_f1': 0,
                    'precision_at_optimal': 0,
                    'recall_at_optimal': 0,
                    'precision_values': [1.0, 0.0],  # Default curve
                    'recall_values': [0.0, 1.0],
                    'thresholds': [1.0, 0.0],
                    'f1_scores': [0.0, 0.0]
                }
                continue
            
            # Calculate precision-recall at each threshold
            thresholds = [p[0] for p in preds]
            tp_cumsum = np.cumsum([p[1] for p in preds])
            fp_cumsum = np.cumsum([1-p[1] for p in preds])
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
            recall = tp_cumsum / (total_gt[class_id] + 1e-10)
            
            # Add endpoints for complete curve
            precision = np.concatenate([[1.0], precision, [0.0]])
            recall = np.concatenate([[0.0], recall, [recall[-1]]])
            thresholds = np.concatenate([[1.0], thresholds, [0.0]])
            
            # Calculate F1 scores
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Calculate AP
            ap = self._calculate_ap(precision[1:-1], recall[1:-1])  # Exclude added endpoints
            
            # Find optimal threshold (max F1)
            valid_f1 = f1_scores[1:-1]  # Exclude endpoints
            if len(valid_f1) > 0:
                best_idx = np.argmax(valid_f1) + 1  # Adjust for added start point
                optimal_thresh = thresholds[best_idx]
                max_f1 = f1_scores[best_idx]
                precision_at_optimal = precision[best_idx]
                recall_at_optimal = recall[best_idx]
            else:
                optimal_thresh = 0.5
                max_f1 = 0
                precision_at_optimal = 0
                recall_at_optimal = 0
            
            pr_results[class_name] = {
                'ap': float(ap),
                'optimal_threshold': float(optimal_thresh),
                'max_f1': float(max_f1),
                'precision_at_optimal': float(precision_at_optimal),
                'recall_at_optimal': float(recall_at_optimal),
                # NEW: Store full curve data for plotting
                'precision_values': precision.tolist(),
                'recall_values': recall.tolist(),
                'thresholds': thresholds.tolist(),
                'f1_scores': f1_scores.tolist()
            }
        
        return pr_results
    
    def compute_stratified_analysis(self, split):
        """Critical: Prevalence-stratified analysis"""
        bins = {
            "0-1%": [],
            "1-3%": [],
            "3-5%": [],
            ">5%": []
        }
        
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        # Only for binary task
        if self.config.task != 'binary':
            return bins
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        for img_path in tqdm(img_files, desc="Stratified analysis"):
            label_path = lbl_dir / (img_path.stem + '.txt')
            
            # Count classes in GT
            class_counts = defaultdict(int)
            gt_boxes_by_class = defaultdict(list)
            
            if label_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            gt_boxes_by_class[class_id].append(
                                self._xywh_to_xyxy([x_center, y_center, width, height])
                            )
            
            # Calculate infected ratio
            infected_count = class_counts[1]  # Class 1 is infected
            total_count = sum(class_counts.values())
            
            if total_count == 0 or infected_count == 0:
                continue
            
            infected_ratio = infected_count / total_count * 100
            
            # Get predictions
            results = self.model.predict(img_path, conf=0.5, iou=0.5, verbose=False)[0]
            
            # Calculate infected recall
            infected_tp = 0
            if results.boxes is not None:
                for box in results.boxes:
                    if int(box.cls.item()) == 1:  # Infected class
                        pred_box = box.xyxy[0].tolist()
                        
                        # Check if matches any infected GT
                        for gt_box in gt_boxes_by_class[1]:
                            if self._compute_iou(pred_box, gt_box) > 0.5:
                                infected_tp += 1
                                break
            
            infected_recall = infected_tp / infected_count if infected_count > 0 else 0
            
            # Bin the result
            if infected_ratio < 1:
                bins["0-1%"].append(infected_recall)
            elif infected_ratio < 3:
                bins["1-3%"].append(infected_recall)
            elif infected_ratio < 5:
                bins["3-5%"].append(infected_recall)
            else:
                bins[">5%"].append(infected_recall)
        
        # Calculate statistics per bin
        bin_stats = {}
        for bin_name, recalls in bins.items():
            if recalls:
                bin_stats[bin_name] = {
                    'mean_recall': np.mean(recalls),
                    'std_recall': np.std(recalls),
                    'median_recall': np.median(recalls),
                    'count': len(recalls)
                }
            else:
                bin_stats[bin_name] = {
                    'mean_recall': 0,
                    'std_recall': 0,
                    'median_recall': 0,
                    'count': 0
                }
        
        return bin_stats
    
    def compute_error_analysis(self, split):
        """TIDE-style error analysis - both per-class and aggregate"""
        # Initialize error tracking
        error_types_aggregate = {
            'classification': [],
            'localization': [],
            'duplicate': [],
            'background': [],
            'missed': []
        }
        
        error_types_per_class = {}
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            error_types_per_class[class_name] = {
                'classification': [],
                'localization': [],
                'duplicate': [],
                'background': [],
                'missed': []
            }
        
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        # FIX 2: Remove [:100] limit - analyze ALL images
        img_files = list(img_dir.glob('*.jpg'))  # Analyze ALL images, not just first 100
        
        for img_path in tqdm(img_files, desc="Error analysis"):
            label_path = lbl_dir / (img_path.stem + '.txt')
            
            # Load GT boxes
            gt_boxes = []
            if label_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            gt_boxes.append({
                                'class_id': class_id,
                                'box': self._xywh_to_xyxy([x_center, y_center, width, height]),
                                'matched': False
                            })
            
            # Get predictions
            results = self.model.predict(img_path, conf=0.5, iou=0.5, verbose=False)[0]
            pred_boxes = []
            
            if results.boxes is not None:
                for box in results.boxes:
                    pred_boxes.append({
                        'class_id': int(box.cls.item()),
                        'box': box.xyxy[0].tolist(),
                        'conf': box.conf.item(),
                        'matched': False
                    })
            
            # Analyze errors for each prediction
            for pred_idx, pred in enumerate(pred_boxes):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt['matched']:
                        continue
                    
                    iou = self._compute_iou(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                pred_class_name = self.class_names[pred['class_id']]
                
                if best_iou >= 0.5:
                    gt = gt_boxes[best_gt_idx]
                    
                    if pred['class_id'] != gt['class_id']:
                        # Classification error
                        error_types_aggregate['classification'].append(1)
                        error_types_per_class[pred_class_name]['classification'].append(1)
                        
                        # Also count for GT class
                        gt_class_name = self.class_names[gt['class_id']]
                        error_types_per_class[gt_class_name]['classification'].append(1)
                    else:
                        # Localization error (IoU between 0.5 and 1.0)
                        error_value = 1 - best_iou
                        error_types_aggregate['localization'].append(error_value)
                        error_types_per_class[pred_class_name]['localization'].append(error_value)
                    
                    gt['matched'] = True
                    pred['matched'] = True
                    
                elif best_iou > 0.1:  # Some overlap but not enough
                    # Background/false positive with some localization error
                    error_types_aggregate['background'].append(1)
                    error_types_per_class[pred_class_name]['background'].append(1)
                else:
                    # Pure background/false positive
                    error_types_aggregate['background'].append(1)
                    error_types_per_class[pred_class_name]['background'].append(1)
            
            # FIX 1: Check for duplicate detections - remove matched checks
            for i, pred1 in enumerate(pred_boxes):
                # No check for matched status - check ALL predictions
                for j, pred2 in enumerate(pred_boxes[i+1:], i+1):
                    # No check for matched status - check ALL predictions
                    if pred1['class_id'] == pred2['class_id']:
                        iou = self._compute_iou(pred1['box'], pred2['box'])
                        if iou > 0.5:
                            error_types_aggregate['duplicate'].append(1)
                            class_name = self.class_names[pred1['class_id']]
                            error_types_per_class[class_name]['duplicate'].append(1)
            
            # Check for missed detections
            for gt in gt_boxes:
                if not gt['matched']:
                    error_types_aggregate['missed'].append(1)
                    class_name = self.class_names[gt['class_id']]
                    error_types_per_class[class_name]['missed'].append(1)
        
        # Calculate statistics for aggregate
        error_stats = {}
        for error_type, values in error_types_aggregate.items():
            if values:
                error_stats[error_type] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values)
                }
            else:
                error_stats[error_type] = {'mean': 0, 'std': 0, 'count': 0}
        
        # Calculate statistics per class
        error_stats_per_class = {}
        for class_name in self.class_names.values():
            error_stats_per_class[class_name] = {}
            for error_type in ['classification', 'localization', 'duplicate', 'background', 'missed']:
                values = error_types_per_class[class_name][error_type]
                if values:
                    error_stats_per_class[class_name][error_type] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                else:
                    error_stats_per_class[class_name][error_type] = {
                        'mean': 0,
                        'std': 0,
                        'count': 0
                    }
        
        # Store per-class errors
        self.errors_per_class = error_stats_per_class
        
        return error_stats
    
    def compute_confusion_matrix(self, split):
        """Object-level confusion matrix"""
        # Initialize confusion matrix
        cm = np.zeros((self.num_classes + 1, self.num_classes + 1))  # +1 for background/missed
        
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        
        # FIX 2: Remove [:100] limit from the loop
        for img_path in tqdm(img_files, desc="Building confusion matrix"):  # Process ALL images
            label_path = lbl_dir / (img_path.stem + '.txt')
            
            # Load GT
            gt_boxes = []
            if label_path.exists():
                img = Image.open(img_path)
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1]) * img_width
                            y_center = float(parts[2]) * img_height
                            width = float(parts[3]) * img_width
                            height = float(parts[4]) * img_height
                            
                            gt_boxes.append({
                                'class_id': class_id,
                                'box': self._xywh_to_xyxy([x_center, y_center, width, height]),
                                'matched': False
                            })
            
            # Get predictions
            results = self.model.predict(img_path, conf=0.5, iou=0.5, verbose=False)[0]
            
            if results.boxes is not None:
                for box in results.boxes:
                    pred_class = int(box.cls.item())
                    pred_box = box.xyxy[0].tolist()
                    
                    # Find best matching GT
                    best_iou = 0.5
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_boxes):
                        if gt['matched']:
                            continue
                        
                        iou = self._compute_iou(pred_box, gt['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_gt_idx >= 0:
                        gt_class = gt_boxes[best_gt_idx]['class_id']
                        cm[gt_class, pred_class] += 1
                        gt_boxes[best_gt_idx]['matched'] = True
                    else:
                        # False positive (background)
                        cm[self.num_classes, pred_class] += 1
            
            # Count missed detections
            for gt in gt_boxes:
                if not gt['matched']:
                    cm[gt['class_id'], self.num_classes] += 1
        
        return cm.tolist()  # Return as list for JSON serialization
    
    def visualize_predictions(self, split='test', num_samples=10, save=True):
        """Visualize model predictions on sample images"""
        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        
        img_files = list(img_dir.glob('*.jpg'))[:num_samples]
        
        fig, axes = plt.subplots(2, min(5, num_samples), figsize=(20, 8))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, img_path in enumerate(img_files):
            row = idx // 5
            col = idx % 5
            
            if idx >= 10:
                break
            
            # Load image
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # Get predictions
            results = self.model.predict(img_path, conf=0.5, iou=0.5, verbose=False)[0]
            
            # Plot
            axes[row, col].imshow(img_array)
            axes[row, col].set_title(f"{img_path.stem[:10]}", fontsize=8)
            axes[row, col].axis('off')
            
            # Draw predictions
            if results.boxes is not None:
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls.item())
                    conf = box.conf.item()
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=self.colors[class_id],
                        facecolor='none'
                    )
                    axes[row, col].add_patch(rect)
                    
                    # Add label
                    axes[row, col].text(
                        x1, y1-5,
                        f"{self.class_names[class_id]}: {conf:.2f}",
                        color='white',
                        fontsize=6,
                        bbox=dict(facecolor=self.colors[class_id], alpha=0.7)
                    )
        
        plt.suptitle(f"Predictions on {split.upper()} Set", fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f"predictions_{split}.png"
            plt.savefig(save_path, dpi=150)
        
        return fig
    
    def save_results(self, results):
        """Save all results to files"""
        # Save JSON
        json_path = self.output_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save readable report
        self.generate_report(results)
    
    def generate_report(self, results):
        """Generate comprehensive text and visual reports"""
        report_path = self.output_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE EVALUATION REPORT\n")
            f.write(f"Dataset: {results['dataset'].upper()} | Task: {results['task']}\n")
            f.write(f"Model: {results['model']} | Split: {results['split'].upper()}\n")
            f.write("="*70 + "\n\n")
            
            # 1. Global Metrics
            f.write("1. GLOBAL METRICS\n")
            f.write("-"*50 + "\n")
            global_table = []
            for metric, value in results['global'].items():
                global_table.append([metric, f"{value:.4f}"])
            f.write(tabulate(global_table, headers=['Metric', 'Value'], tablefmt='grid'))
            f.write("\n\n")
            
            # 2. Per-Class Metrics
            f.write("2. PER-CLASS METRICS\n")
            f.write("-"*50 + "\n")
            class_table = []
            for class_name, metrics in results['per_class'].items():
                class_table.append([
                    class_name,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1_score']:.3f}",
                    metrics['support']
                ])
            f.write(tabulate(class_table, 
                           headers=['Class', 'Precision', 'Recall', 'F1', 'Support'],
                           tablefmt='grid'))
            f.write("\n\n")
            
            # 3. PR Analysis
            f.write("3. PRECISION-RECALL ANALYSIS\n")
            f.write("-"*50 + "\n")
            pr_table = []
            for class_name, pr_stats in results.get('pr_analysis', {}).items():
                pr_table.append([
                    class_name,
                    f"{pr_stats.get('ap', 0):.3f}",
                    f"{pr_stats.get('optimal_threshold', 0):.3f}",
                    f"{pr_stats.get('max_f1', 0):.3f}"
                ])
            f.write(tabulate(pr_table,
                           headers=['Class', 'AP', 'Optimal Threshold', 'Max F1'],
                           tablefmt='grid'))
            f.write("\n\n")
            
            # 4. Stratified Analysis (CRITICAL)
            if self.config.task == 'binary':
                f.write("4. PREVALENCE-STRATIFIED ANALYSIS (Critical for Clinical Use)\n")
                f.write("-"*50 + "\n")
                strat_table = []
                for bin_name, stats in results.get('stratified', {}).items():
                    strat_table.append([
                        bin_name,
                        f"{stats.get('mean_recall', 0):.3f}",
                        f"{stats.get('std_recall', 0):.3f}",
                        stats.get('count', 0)
                    ])
                f.write(tabulate(strat_table,
                               headers=['Density Bin', 'Mean Recall', 'Std Dev', 'Count'],
                               tablefmt='grid'))
                f.write("\n\n")
            
            # 5. Error Analysis
            f.write("5. ERROR ANALYSIS (TIDE)\n")
            f.write("-"*50 + "\n")
            error_table = []
            for error_type, stats in results.get('errors', {}).items():
                error_table.append([
                    error_type.capitalize(),
                    f"{stats.get('mean', 0):.3f}",
                    stats.get('count', 0)
                ])
            f.write(tabulate(error_table,
                           headers=['Error Type', 'Mean Rate', 'Count'],
                           tablefmt='grid'))
        
        print(f"\nReport saved to: {report_path}")
    
    # Helper methods
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _xywh_to_xyxy(self, box):
        """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
        x_center, y_center, w, h = box
        return [x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]
    
    def _calculate_ap(self, precision, recall):
        """Calculate Average Precision using 11-point interpolation"""
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        return ap