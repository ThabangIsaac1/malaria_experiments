#!/usr/bin/env python3
"""
Smoke test for stratified analysis fix
Tests that recall values are now ≤ 1.0 after fixing duplicate matching bug
"""

from pathlib import Path
from src.evaluation.evaluator import ComprehensiveEvaluator
import sys

# Configuration
class Config:
    model_name = 'rtdetr'
    conf = 0.25
    iou = 0.45
    task = 'binary'
    dataset = 'd1'

    def get_class_names(self):
        return {0: 'Uninfected', 1: 'Infected'}

config = Config()

# Paths
base_dir = Path(__file__).parent.parent
dataset_path = base_dir / "dataset_d1/yolo_format/binary"

# Use the rtdetr_200.pt model in qgfl_experiments directory
model_path = Path(__file__).parent / "rtdetr_200.pt"

if not model_path.exists():
    print(f"❌ ERROR: Model not found at {model_path}")
    print("\nPlease ensure rtdetr_200.pt is in the qgfl_experiments directory.")
    sys.exit(1)

print("="*70)
print("STRATIFIED ANALYSIS FIX - SMOKE TEST")
print("="*70)
print(f"Model: {model_path.name}")
print(f"Dataset: {dataset_path}")
print(f"Task: binary (required for stratified analysis)")
print(f"Thresholds: conf={config.conf}, iou={config.iou}")
print("="*70)
print()

# Initialize evaluator
print("[1/3] Initializing evaluator...")
try:
    evaluator = ComprehensiveEvaluator(
        model_path=model_path,
        dataset_path=dataset_path,
        config=config,
        output_dir="test_stratified_fix_output",
        yaml_path=Path(__file__).parent / f"configs/data_yamls/{config.dataset}_{config.task}.yaml"
    )
    print(f"✅ Evaluator initialized")
    print(f"   Model type: {type(evaluator.model).__name__}")
except Exception as e:
    print(f"❌ FAILED to initialize: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Run stratified analysis
print("[2/3] Running stratified analysis on test set...")
print("   (This will process all test images to compute prevalence bins)")
try:
    stratified_results = evaluator.compute_stratified_analysis(split='test')
    print("✅ Stratified analysis completed")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Validate results
print("[3/3] Validating results (checking for recall > 1.0 bug)...")
print()
print("Results:")
print("="*70)
print(f"{'Parasitemia Bin':<20} {'Mean Recall':<15} {'Std Recall':<15} {'Count':<10} {'Status'}")
print("="*70)

all_valid = True
for bin_name, stats in stratified_results.items():
    mean_recall = stats['mean_recall']
    std_recall = stats['std_recall']
    count = stats['count']

    # Check if recall is valid (≤ 1.0)
    if mean_recall > 1.0:
        status = "❌ INVALID (>1.0)"
        all_valid = False
    else:
        status = "✅ VALID"

    print(f"{bin_name:<20} {mean_recall:<15.4f} {std_recall:<15.4f} {count:<10} {status}")

print("="*70)
print()

# Final verdict
if all_valid:
    print("🎉 SUCCESS: All recall values are ≤ 1.0")
    print("   The duplicate matching bug has been FIXED!")
    print()
    print("   Details:")
    for bin_name, stats in stratified_results.items():
        if stats['count'] > 0:
            print(f"   {bin_name}: {stats['mean_recall']:.1%} recall on {stats['count']} images")
    sys.exit(0)
else:
    print("❌ FAILURE: Some recall values exceed 1.0")
    print("   The bug is still present - investigation needed!")
    sys.exit(1)
