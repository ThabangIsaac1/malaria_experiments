"""
Species-Specific Evaluator for Multi-Species Plasmodium Detection

Extends ComprehensiveEvaluator with species-specific analysis methods
for D3 dataset (5 classes: Uninfected + 4 Plasmodium species).

Key additions:
1. compute_species_stratified_analysis() - Per-species recall breakdown
2. Rare species average computation (P. ovale, P. malariae, P. vivax)
3. Species-optimized evaluation for Guemas et al. 2024 comparison

Author: Thabang Isaka
Date: 2025-12-05
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any, List

# Import base evaluator
from .evaluator import ComprehensiveEvaluator


# Species configuration for D3 dataset
SPECIES_CLASS_NAMES = {
    0: 'Uninfected',
    1: 'P_falciparum',
    2: 'P_ovale',
    3: 'P_malariae',
    4: 'P_vivax',
}

RARE_SPECIES = ['P_ovale', 'P_malariae', 'P_vivax']
PARASITE_CLASSES = [1, 2, 3, 4]  # All species except uninfected


class SpeciesEvaluator(ComprehensiveEvaluator):
    """
    Extended evaluator for species-level Plasmodium detection

    Inherits all methods from ComprehensiveEvaluator and adds:
    - compute_species_stratified_analysis() for per-species breakdown
    - Rare species average computation
    - Species confusion analysis
    """

    def __init__(self, model_path, dataset_path, config, output_dir="evaluation_results", yaml_path=None):
        """
        Initialize species evaluator

        Args:
            model_path: Path to trained model
            dataset_path: Path to dataset
            config: Configuration object with class names, thresholds, etc.
            output_dir: Directory for saving results
            yaml_path: Path to dataset YAML file
        """
        super().__init__(model_path, dataset_path, config, output_dir, yaml_path)

        # Ensure we have species class names
        if self.num_classes != 5:
            print(f"[WARNING] Expected 5 classes for species detection, got {self.num_classes}")

    def run_full_evaluation(self, split='test'):
        """
        Run complete evaluation pipeline with species-specific analysis

        Extends base evaluation with:
        - Species stratified analysis (per-species metrics)
        - Rare species average
        - Rare species stratified analysis (rare species at different density levels)
        """
        # Run base evaluation
        results = super().run_full_evaluation(split)

        # Add species-specific analysis
        print("\n[7/8] Computing species stratified analysis...")
        results['species_analysis'] = self.compute_species_stratified_analysis(split)

        # Add rare species stratified analysis (WHO-aligned, SA-QGFL contribution)
        print("\n[8/8] Computing rare species stratified analysis...")
        results['rare_species_stratified'] = self.compute_rare_species_stratified(split)

        # Save updated results
        self.save_results(results)

        return results

    def compute_species_stratified_analysis(self, split: str) -> Dict[str, Any]:
        """
        Per-species metrics breakdown for Guemas comparison

        Computes:
        - Per-species precision, recall, F1, mAP
        - Rare species average (P. ovale, P. malariae, P. vivax)
        - Species-specific support counts

        Args:
            split: Data split to evaluate ('train', 'val', or 'test')

        Returns:
            Dictionary with per-species metrics and rare species average
        """
        species_stats = {}

        # Get per-class metrics (already computed by base class)
        per_class = self.compute_per_class_metrics(split)
        global_metrics = self.compute_global_metrics(split)

        # Extract metrics for each parasite species (skip Uninfected)
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]

            if class_name in per_class:
                metrics = per_class[class_name]
                species_stats[class_name] = {
                    'recall': metrics.get('recall', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'f1_score': metrics.get('f1_score', 0.0),
                    'support': metrics.get('support', 0),
                    'tp': metrics.get('tp', 0),
                    'fp': metrics.get('fp', 0),
                    'fn': metrics.get('fn', 0),
                    'mAP50_95': global_metrics.get('per_class_map', {}).get(class_name, 0.0),
                }
            else:
                species_stats[class_name] = {
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1_score': 0.0,
                    'support': 0,
                    'tp': 0,
                    'fp': 0,
                    'fn': 0,
                    'mAP50_95': 0.0,
                }

        # Compute rare species average (Guemas comparison metric)
        rare_recalls = []
        rare_precisions = []
        rare_f1s = []

        for species in RARE_SPECIES:
            if species in species_stats:
                rare_recalls.append(species_stats[species]['recall'])
                rare_precisions.append(species_stats[species]['precision'])
                rare_f1s.append(species_stats[species]['f1_score'])

        species_stats['rare_species_metrics'] = {
            'avg_recall': np.mean(rare_recalls) if rare_recalls else 0.0,
            'avg_precision': np.mean(rare_precisions) if rare_precisions else 0.0,
            'avg_f1': np.mean(rare_f1s) if rare_f1s else 0.0,
            'species_included': RARE_SPECIES,
        }

        # Compute overall parasite metrics (all species vs uninfected)
        parasite_tp = sum(species_stats.get(self.class_names.get(c, f'Class_{c}'), {}).get('tp', 0)
                        for c in PARASITE_CLASSES if c < self.num_classes)
        parasite_fp = sum(species_stats.get(self.class_names.get(c, f'Class_{c}'), {}).get('fp', 0)
                        for c in PARASITE_CLASSES if c < self.num_classes)
        parasite_fn = sum(species_stats.get(self.class_names.get(c, f'Class_{c}'), {}).get('fn', 0)
                        for c in PARASITE_CLASSES if c < self.num_classes)

        parasite_precision = parasite_tp / (parasite_tp + parasite_fp) if (parasite_tp + parasite_fp) > 0 else 0
        parasite_recall = parasite_tp / (parasite_tp + parasite_fn) if (parasite_tp + parasite_fn) > 0 else 0
        parasite_f1 = 2 * parasite_precision * parasite_recall / (parasite_precision + parasite_recall) if (parasite_precision + parasite_recall) > 0 else 0

        species_stats['all_parasites'] = {
            'precision': parasite_precision,
            'recall': parasite_recall,
            'f1_score': parasite_f1,
            'tp': parasite_tp,
            'fp': parasite_fp,
            'fn': parasite_fn,
        }

        return species_stats

    def compute_species_confusion_analysis(self, split: str) -> Dict[str, Any]:
        """
        Analyze species confusion patterns

        Identifies which species are commonly confused with each other,
        which is important for understanding model behavior on rare species.

        Args:
            split: Data split to evaluate

        Returns:
            Dictionary with confusion patterns
        """
        # Get confusion matrix from base class
        cm = np.array(self.compute_confusion_matrix(split))

        confusion_analysis = {}

        # For each species, find what it's most confused with
        for class_id in range(1, self.num_classes):
            class_name = self.class_names[class_id]
            row = cm[class_id, :]  # GT = this class

            # Get confusion counts (excluding correct predictions and background)
            confusions = {}
            for other_id in range(self.num_classes):
                if other_id != class_id:
                    other_name = self.class_names[other_id]
                    confusions[other_name] = int(row[other_id])

            # Sort by confusion count
            sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)

            confusion_analysis[class_name] = {
                'total_gt': int(row.sum()),
                'correct': int(cm[class_id, class_id]),
                'missed': int(cm[class_id, self.num_classes]) if self.num_classes < cm.shape[1] else 0,
                'top_confusions': sorted_confusions[:3],  # Top 3 confused classes
            }

        return confusion_analysis

    def compute_rare_species_stratified(self, split: str) -> Dict[str, Any]:
        """
        Rare species detection at different total parasitemia levels

        WHO-aligned analysis showing SA-QGFL's unique contribution:
        - Filter to images containing rare species (P. ovale, P. malariae, P. vivax)
        - Bin by total parasitemia (0-1%, 1-3%, 3-5%, >5%)
        - Measure rare species recall at each bin

        This demonstrates SA-QGFL's value for minority class detection
        at clinically relevant parasite density levels.

        Args:
            split: Data split to evaluate ('train', 'val', or 'test')

        Returns:
            Dictionary with bin statistics for rare species recall
        """
        from PIL import Image

        RARE_SPECIES_IDS = [2, 3, 4]  # P_ovale, P_malariae, P_vivax
        ALL_PARASITE_IDS = [1, 2, 3, 4]

        bins = {"0-1%": [], "1-3%": [], "3-5%": [], ">5%": []}

        img_dir = self.dataset_path / split / "images"
        lbl_dir = self.dataset_path / split / "labels"
        img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))

        images_with_rare = 0

        for img_path in tqdm(img_files, desc="Rare species stratified"):
            label_path = lbl_dir / (img_path.stem + '.txt')

            # Load GT boxes
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

            # Only process images with rare species
            rare_count = sum(class_counts[c] for c in RARE_SPECIES_IDS)
            if rare_count == 0:
                continue  # Skip images without rare species

            images_with_rare += 1

            # Calculate total parasitemia (all parasites / all cells)
            total_parasites = sum(class_counts[c] for c in ALL_PARASITE_IDS)
            total_cells = sum(class_counts.values())
            parasitemia = total_parasites / total_cells * 100 if total_cells > 0 else 0

            # Get predictions
            results = self.model.predict(
                img_path, conf=self.config.conf, iou=self.config.iou,
                agnostic_nms=True, verbose=False
            )[0]

            # Compute rare species recall
            rare_tp = 0
            matched_gt = {c: [False] * len(gt_boxes_by_class[c]) for c in RARE_SPECIES_IDS}

            if results.boxes is not None:
                for box in results.boxes:
                    pred_class = int(box.cls.item())
                    if pred_class in RARE_SPECIES_IDS:
                        pred_box = box.xyxy[0].tolist()

                        # Match to GT of same class
                        for gt_idx, gt_box in enumerate(gt_boxes_by_class[pred_class]):
                            if not matched_gt[pred_class][gt_idx]:
                                if self._compute_iou(pred_box, gt_box) > 0.45:
                                    rare_tp += 1
                                    matched_gt[pred_class][gt_idx] = True
                                    break

            rare_recall = rare_tp / rare_count if rare_count > 0 else 0

            # Bin by total parasitemia
            if parasitemia < 1:
                bins["0-1%"].append(rare_recall)
            elif parasitemia < 3:
                bins["1-3%"].append(rare_recall)
            elif parasitemia < 5:
                bins["3-5%"].append(rare_recall)
            else:
                bins[">5%"].append(rare_recall)

        # Calculate statistics per bin
        bin_stats = {}
        for bin_name, recalls in bins.items():
            if recalls:
                bin_stats[bin_name] = {
                    'mean_recall': float(np.mean(recalls)),
                    'std_recall': float(np.std(recalls)),
                    'median_recall': float(np.median(recalls)),
                    'count': len(recalls)
                }
            else:
                bin_stats[bin_name] = {
                    'mean_recall': 0,
                    'std_recall': 0,
                    'median_recall': 0,
                    'count': 0
                }

        # Add summary
        bin_stats['_summary'] = {
            'total_images_with_rare': images_with_rare,
            'rare_species': ['P_ovale', 'P_malariae', 'P_vivax']
        }

        return bin_stats

    def print_species_summary(self, results: Dict[str, Any]):
        """
        Print formatted species detection summary

        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*70)
        print("SPECIES DETECTION SUMMARY")
        print("="*70)

        if 'species_analysis' in results:
            species = results['species_analysis']

            # Print per-species table
            print("\nPer-Species Performance:")
            print("-" * 60)
            print(f"{'Species':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
            print("-" * 60)

            for class_name in ['P_falciparum', 'P_ovale', 'P_malariae', 'P_vivax']:
                if class_name in species:
                    s = species[class_name]
                    print(f"{class_name:<15} {s['precision']:.4f}       {s['recall']:.4f}       {s['f1_score']:.4f}       {s['support']}")

            print("-" * 60)

            # Print rare species average
            if 'rare_species_metrics' in species:
                rare = species['rare_species_metrics']
                print(f"\nRare Species Average (P. ovale, P. malariae, P. vivax):")
                print(f"  Recall:    {rare['avg_recall']:.4f}")
                print(f"  Precision: {rare['avg_precision']:.4f}")
                print(f"  F1:        {rare['avg_f1']:.4f}")

            # Print overall parasite metrics
            if 'all_parasites' in species:
                parasites = species['all_parasites']
                print(f"\nAll Parasites Combined:")
                print(f"  Recall:    {parasites['recall']:.4f}")
                print(f"  Precision: {parasites['precision']:.4f}")
                print(f"  F1:        {parasites['f1_score']:.4f}")

        print("="*70)


def create_species_evaluator(
    model_path,
    dataset_path,
    config,
    output_dir="evaluation_results",
    yaml_path=None
) -> SpeciesEvaluator:
    """
    Factory function to create species evaluator

    Args:
        model_path: Path to trained model
        dataset_path: Path to dataset
        config: Configuration object
        output_dir: Directory for results
        yaml_path: Path to dataset YAML

    Returns:
        Configured SpeciesEvaluator instance
    """
    return SpeciesEvaluator(
        model_path=model_path,
        dataset_path=dataset_path,
        config=config,
        output_dir=output_dir,
        yaml_path=yaml_path
    )


# Standalone testing
if __name__ == "__main__":
    print("Species Evaluator module loaded successfully")
    print(f"Configured for {len(SPECIES_CLASS_NAMES)} classes:")
    for class_id, name in SPECIES_CLASS_NAMES.items():
        print(f"  {class_id}: {name}")
    print(f"\nRare species for averaging: {RARE_SPECIES}")
