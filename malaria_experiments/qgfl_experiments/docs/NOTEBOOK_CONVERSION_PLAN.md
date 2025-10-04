# Notebook Conversion Plan
## Converting 01_run_baseline.ipynb → cluster_run_baseline.py

**Date:** 2025-10-03
**Purpose:** Create cluster-ready Python script from working notebook

---

## Current Notebook Status

**File:** `01_run_baseline.ipynb`
**Total Lines:** 3,196 (when converted to .py)
**Status:** ✅ Complete and functional

### Cells Present (19 cells):
1. ✅ Cell 1: Setup (imports)
2. ✅ Cell 2: Configuration (ExperimentConfig)
3. ✅ Cell 3: Prepare Dataset
4. ✅ Cell 4: Create YAML + Dataset Visualization
5. ✅ Cell 5: Initialize Model
6. ✅ Cell 7: W&B Initialization
7. ✅ Cell 8: Complete Training + Post-Training Logging
8. ✅ Cell 8.5: Inference Time Tracking
9. ✅ Cell 10: Initialize Evaluator
10. ✅ Cell 11: Global Metrics - Validation
11. ✅ Cell 12: Global Metrics - Test
12. ✅ Cell 13A: Per-Class Analysis - Validation
13. ✅ Cell 13B: Per-Class Analysis - Test
14. ✅ Cell 14: Recall Variability Analysis
15. ✅ Cell 14: Prevalence-Stratified Analysis
16. ✅ Cell 15: Precision-Recall Curves
17. ✅ Cell 16: TIDE Error Analysis
18. ✅ Cell 17: Ground Truth vs Predictions
19. ✅ Cell 18: Decision Analysis + Summary
20. ✅ Cell 19: Comprehensive W&B Logging (with model upload!)

---

## What Needs to Change for Cluster

### 1. Hardcoded Values → Command-Line Arguments

**Current (Cell 2):**
```python
config = ExperimentConfig(
    dataset='d3',        # HARDCODED
    model_name='yolo11s', # HARDCODED
    task='binary',
    # ... other params
)
```

**Needed:**
```python
# Add argparse at top
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['d1', 'd2', 'd3'], required=True)
parser.add_argument('--model', choices=['yolov8s', 'yolo11s'], required=True)
parser.add_argument('--task', default='binary')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--use-wandb', action='store_true', default=True)
args = parser.parse_args()

config = ExperimentConfig(
    dataset=args.dataset,
    model_name=args.model,
    task=args.task,
    epochs=args.epochs,
    batch_size=args.batch_size,
    use_wandb=args.use_wandb,
)
```

### 2. File Paths → Absolute/Configurable

**Current:**
```python
yaml_path = yaml_dir / f'{config.dataset}_{config.task}.yaml'
```

**Check:** Ensure paths work when script runs from different locations

### 3. Model Files Location

**Current:**
```python
if config.model_name == 'yolov8s':
    model = YOLO('yolov8s.pt')  # Looks in current dir
```

**Needed:**
```python
# Define model paths (cluster might need full path)
model_path = Path('path/to/models') / f'{config.model_name}.pt'
model = YOLO(str(model_path))
```

### 4. W&B API Key Handling

**Current (Cell 2):**
```python
config = ExperimentConfig(
    wandb_key='...',  # Hardcoded or in config
)
```

**Needed:**
```python
# Use environment variable for cluster
wandb_key = os.getenv('WANDB_API_KEY', config.wandb_key)
wandb.login(key=wandb_key)
```

### 5. Results Directory

**Current:**
```python
results_dir = Path('../results') / experiment_name
```

**Cluster:**
```python
# Configurable base directory
results_base = Path(args.results_dir) if args.results_dir else Path('../results')
results_dir = results_base / experiment_name
```

---

## Conversion Strategy

### Option A: Minimal Changes (Recommended)
1. Add argparse at top
2. Replace hardcoded values with `args.*`
3. Keep ALL evaluation logic as-is
4. Test locally before cluster

### Option B: Full Refactor
1. Convert to functions
2. Add error handling
3. Modularize components
4. More work, higher risk

**Recommendation:** Option A - minimal changes to proven working code

---

## Conversion Steps

1. **Export notebook to .py:**
   ```bash
   jupyter nbconvert --to python 01_run_baseline.ipynb --output cluster_run_baseline.py
   ```

2. **Add argparse section (after imports):**
   - Parse: dataset, model, task, epochs, batch_size, use_wandb
   - Add: results_dir, wandb_key (optional)

3. **Update Cell 2 (Configuration):**
   - Replace hardcoded values with `args.*`

4. **Test locally:**
   ```bash
   python cluster_run_baseline.py --dataset d1 --model yolov8s --epochs 2
   ```

5. **Verify outputs:**
   - Check results directory created
   - Check W&B logging works
   - Check all visualizations generated

6. **Create cluster submission script:**
   - SLURM script that calls with different params
   - 6 jobs: 2 models × 3 datasets

---

## Cluster-Specific Considerations

### Need to Know:
1. **Cluster type:** SLURM? PBS? Other?
2. **Python environment:** Conda? Virtualenv? Modules?
3. **GPU allocation:** How to request GPUs?
4. **Working directory:** Where will script run from?
5. **Data location:** Where are datasets stored on cluster?
6. **Results storage:** Where to save outputs?
7. **W&B access:** Internet access available?

### Typical SLURM Script:
```bash
#!/bin/bash
#SBATCH --job-name=yolo_d1_v8s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load python/3.9
source activate yolo_env

python cluster_run_baseline.py \
    --dataset d1 \
    --model yolov8s \
    --epochs 200 \
    --batch-size 16 \
    --use-wandb
```

---

## What NOT to Change

❌ **DO NOT modify:**
- Evaluation logic (all cells 10-19)
- Augmentation settings
- Loss function configurations
- W&B logging structure (Cell 19)
- Model training code (Cell 8)

✅ **ONLY change:**
- Hardcoded config values → argparse
- File paths (if needed for cluster)
- Environment-specific settings

---

## Testing Checklist

Before deploying to cluster:

- [ ] Script runs without errors locally
- [ ] Results directory created correctly
- [ ] W&B logging works
- [ ] All visualizations generated:
  - [ ] Training curves
  - [ ] Per-class analysis
  - [ ] Prevalence-stratified
  - [ ] PR curves
  - [ ] TIDE errors
  - [ ] Confusion matrix
  - [ ] Decision analysis
- [ ] Model weights saved (best.pt, last.pt)
- [ ] Model artifact uploaded to W&B
- [ ] Can switch datasets (d1, d2, d3) via args
- [ ] Can switch models (yolov8s, yolov11s) via args

---

## File Structure After Conversion

```
qgfl_experiments/
├── cluster_run_baseline.py      # NEW - Converted script
├── notebooks/
│   └── 01_run_baseline.ipynb    # ORIGINAL - Keep as reference
├── cluster_scripts/             # NEW - SLURM scripts
│   ├── run_d1_yolov8s.sh
│   ├── run_d1_yolov11s.sh
│   ├── run_d2_yolov8s.sh
│   ├── run_d2_yolov11s.sh
│   ├── run_d3_yolov8s.sh
│   └── run_d3_yolov11s.sh
├── configs/
├── src/
└── docs/
```

---

## Next Steps

**Waiting for cluster info:**
1. Cluster type (SLURM/PBS/etc)
2. Environment setup commands
3. GPU request format
4. Data/results paths on cluster
5. W&B access method

**Once received:**
1. Convert notebook → cluster_run_baseline.py
2. Test locally with 2 epochs
3. Create cluster submission scripts
4. Deploy and run 6 baseline experiments

---

**Status:** ✅ Notebook verified complete - Ready for conversion when cluster info provided
