# 01_run_baseline.ipynb - Complete Analysis

## Overview
- **Total lines:** 3,171 (converted to Python)
- **Purpose:** End-to-end training pipeline with W&B logging and comprehensive evaluation
- **Current state:** Fully functional, manually configured per experiment

---

## Workflow Structure

### Phase 1: Setup (Cells 1-2)
```python
Cell 1: Imports, visualization setup, seed setting
Cell 2: Configuration via ExperimentConfig
  - dataset = 'd1', 'd2', or 'd3'  ← MANUALLY CHANGED
  - task = 'binary', 'species', or 'staging'
  - model_name = 'yolov8s' or 'yolov11s'  ← MANUALLY CHANGED
  - epochs = 10 (local) or 200 (cluster)
```

### Phase 2: Dataset Preparation (Cells 3-4)
```python
Cell 3: Auto-convert COCO→YOLO (via utils/coco_to_yolo.py)
Cell 4: Generate YAML config
  - Creates: configs/data_yamls/{dataset}_{task}.yaml
  - Points to: dataset_{dataset}/yolo_format/{task}/
```

### Phase 3: Visualization (Cells 4-5)
```python
Cell 4: Dataset stats and sample visualization
Cell 5: Load pretrained YOLO model
```

### Phase 4: Training (Cells 7-8)
```python
Cell 7: W&B initialization
  - project: malaria_qgfl_experiments
  - name: {model}_{dataset}_{task}_{strategy}

Cell 8: Training execution
  - Gets class distribution
  - Creates training strategy (no_weights/qgfl)
  - Runs model.train(**train_args)

  KEY: train_args configuration
    'project': '../runs/detect'  # Where YOLO saves
    'name': experiment_name      # Subfolder name
    Results saved to: ../runs/detect/{experiment_name}/
```

### Phase 5: Evaluation (Cells 10-19)
```python
Cell 10: Initialize ComprehensiveEvaluator
Cell 11-12: Global metrics (val + test)
Cell 13: Per-class performance
Cell 14: Prevalence-stratified analysis
Cell 15: PR curve visualization
Cell 16: TIDE error analysis
Cell 17: GT vs predictions visualization
Cell 18: Decision analysis
Cell 19: W&B comprehensive logging
```

---

## Critical Findings

### ✅ What Works Well
1. **Modular utilities** - paths.py, coco_to_yolo.py, strategy_wrapper.py all excellent
2. **Comprehensive evaluation** - evaluator.py is publication-ready (1,759 lines)
3. **W&B integration** - Thorough logging of all metrics
4. **Strategy pattern** - Clean separation between NoWeights/QGFL

### ⚠️ Collision Risks

#### 1. **Directory Naming Collision**
**Problem:**
```python
experiment_name = f"{model_name}_{dataset}_{task}_{strategy}"
# Example: "yolov8s_d1_binary_no_weights"

# If you run same experiment twice:
# Run 1: ../runs/detect/yolov8s_d1_binary_no_weights/
# Run 2: ../runs/detect/yolov8s_d1_binary_no_weights/  ← OVERWRITES!
```

**Current behavior:**
- `'exist_ok': True` in train_args allows overwriting
- No timestamp in folder name
- Manual runs could collide with cluster runs

#### 2. **W&B Run Naming**
**Problem:**
```python
wandb.run.name = experiment_name
# Multiple runs with same name create "experiment_name", "experiment_name-1", "experiment_name-2"
# Hard to track which is which
```

#### 3. **Model/Dataset Switching**
**Current:** Manual editing of config in Cell 2
```python
config = ExperimentConfig(
    dataset='d1',        # ← MANUAL CHANGE
    model_name='yolov8s', # ← MANUAL CHANGE
    epochs=10
)
```

**Problem for cluster:**
- Need to edit notebook for each of 6 experiments
- Easy to make mistakes
- Not reproducible

---

## Directory Structure (Current)

```
qgfl_experiments/
├── runs/
│   └── detect/
│       └── {experiment_name}/  ← COLLISION POINT
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           ├── results.csv
│           └── [plots]/
│
├── results/  ← Created by notebook for custom outputs
│   └── {experiment_name}/
│       ├── predictions/
│       └── analysis/
│
└── notebooks/
    └── wandb/  ← W&B local logs
        └── run-{timestamp}-{id}/
```

---

## Recommended Fixes for Cluster

### 1. **Add Timestamp to Experiment Names**
```python
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f"{model}_{dataset}_{task}_{strategy}_{timestamp}"
# Example: "yolov8s_d1_binary_no_weights_20251003_013845"
```

### 2. **Command-Line Arguments**
Convert notebook config to argparse:
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['yolov8s', 'yolov11s'], required=True)
parser.add_argument('--dataset', choices=['d1', 'd2', 'd3'], required=True)
parser.add_argument('--task', choices=['binary', 'species', 'staging'], default='binary')
parser.add_argument('--strategy', choices=['no_weights', 'qgfl'], default='no_weights')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=16)
```

### 3. **Unique Results Directory**
```python
# Instead of: ../runs/detect/{experiment_name}/
# Use:        ../runs/detect/{timestamp}_{experiment_name}/

results_base = Path('../runs/detect')
run_dir = results_base / f"{timestamp}_{experiment_name}"
```

### 4. **W&B Run ID Tracking**
```python
wandb_id = wandb.util.generate_id()
run = wandb.init(
    project=config.wandb_project,
    name=experiment_name,
    id=wandb_id,  # Allows resume
    resume='allow'
)

# Save ID to file for later reference
with open(run_dir / 'wandb_id.txt', 'w') as f:
    f.write(wandb_id)
```

---

## Files That Need to be Called (Dependencies)

### Python Modules (Relative imports from notebook)
```python
sys.path.append('..')  # To access src/

from configs.baseline_config import ExperimentConfig
from src.utils.paths import get_dataset_paths, verify_dataset
from src.utils.visualizer import YOLOVisualizer
from src.evaluation.evaluator import ComprehensiveEvaluator
from src.training.strategy_wrapper import create_training_strategy
```

### External Libraries
```python
ultralytics  # YOLO
wandb
matplotlib, seaborn
pandas, numpy
PIL, tqdm
torch
```

### Data Files
```python
# YAML configs (auto-generated or pre-existing)
configs/data_yamls/{dataset}_{task}.yaml

# Dataset paths (via coco_to_yolo.py)
dataset_{dataset}/yolo_format/{task}/
  ├── train/images/ & labels/
  ├── val/images/ & labels/
  └── test/images/ & labels/
```

---

## Cluster Conversion Strategy

### Option A: Minimal Changes (Keep Notebook Structure)
1. Add argparse at top
2. Replace manual config with args
3. Add timestamp to experiment_name
4. Run with: `python train_baseline.py --model yolov8s --dataset d1`

### Option B: Clean Modular Script (Recommended)
1. Extract core training function
2. Separate evaluation into optional step
3. Create standalone `train.py` and `evaluate.py`
4. Better for parallel cluster jobs

---

## Conclusion

**The notebook is well-structured and production-ready** with these modifications needed:

1. ✅ **Dataset conversion** - Already modular via utils
2. ✅ **Training strategy** - Already modular via strategy_wrapper.py
3. ✅ **Evaluation** - Already modular via evaluator.py
4. ⚠️ **Experiment naming** - Needs timestamp
5. ⚠️ **CLI arguments** - Needs argparse
6. ⚠️ **Directory safety** - Needs collision prevention

**Next:** Create `train_baseline.py` script with all fixes applied.
