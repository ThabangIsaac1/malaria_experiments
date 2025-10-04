# Structured Experiment Execution Plan
**Date**: October 4, 2025
**Status**: Phase 1 Complete (YOLO Baselines), Planning Phase 2

---

## Overview: When Does Each Component Come In?

```
Timeline View:
┌─────────────────────────────────────────────────────────────────┐
│ Week 1-2:  YOLO Baselines (DONE ✅)                            │
│            └─ YOLOv8s/v11s × D1/D2/D3 (binary only)            │
├─────────────────────────────────────────────────────────────────┤
│ Week 3:    RT-DETR Baselines (NEXT ⚡)                          │
│            └─ RT-DETR × D1/D2/D3 (binary only)                 │
├─────────────────────────────────────────────────────────────────┤
│ Week 4-8:  QGFL Progressive (Binary Task)                       │
│            ├─ YOLO + QGFL (5 levels × 3 datasets)              │
│            └─ RT-DETR + Best QGFL (3 experiments)              │
├─────────────────────────────────────────────────────────────────┤
│ Week 9-10: Foundation Models (Binary Task)                      │
│            └─ RedDino + QGFL × D1/D2/D3                        │
├─────────────────────────────────────────────────────────────────┤
│ Week 11-14: Hierarchical Tasks (Species + Staging) ⭐           │
│            ├─ D1/D2: Staging (rings/trophs/schiz)             │
│            ├─ D3: Species classification                        │
│            └─ Best models from binary + QGFL                   │
├─────────────────────────────────────────────────────────────────┤
│ Week 15-16: Ensemble & Final Validation                         │
│            └─ Multi-model ensemble for deployment              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task Progression Strategy

### **Binary Task First (Weeks 1-10)**
**Why:** Foundation for all other tasks

```
Binary Detection (Infected vs Uninfected):
├─ Simplest task - establish baseline
├─ QGFL designed for binary imbalance
├─ Most training data available
├─ Clinical relevance: screening
└─ Paper's original focus
```

### **Hierarchical Tasks Second (Weeks 11-14)** ⭐
**Why:** Build on binary success

```
Staging Task (D1, D2):
├─ Rings, Trophozoites, Schizonts, Gametocytes
├─ Uses same datasets (just different labels)
├─ Clinical relevance: treatment planning
├─ Transfer learning from binary models
└─ QGFL adapts to multi-class

Species Task (D3):
├─ P. falciparum, P. vivax, P. ovale, P. malariae
├─ Multi-species dataset
├─ Clinical relevance: diagnosis
├─ Foundation models (RedDino) critical here
└─ Most challenging task
```

### **Why This Order?**

✅ **Binary → Staging/Species is correct progression**
- Binary: 2 classes, easier, more data
- Staging: 4-5 classes, harder, same datasets
- Species: 4 classes, hardest, domain complexity

✅ **QGFL development on binary first**
- Tune hyperparameters on simpler task
- Validate approach before scaling up
- Transfer learning to hierarchical tasks

✅ **Foundation models last**
- Most expensive experiments
- Need to know optimal QGFL configuration first
- RedDino helps most with complex tasks (species)

---

## Phase 2: RT-DETR Baselines (Week 3) - NEXT STEP

### **Why RT-DETR Now?**

1. **Baseline comparison needed**: Before committing to 30 QGFL experiments
2. **Efficient resource use**: Run while analyzing YOLO results
3. **Informed QGFL strategy**: Know which architecture benefits most
4. **Paper requirement**: Need architecture comparison section

### **RT-DETR Code Structure**

**Good news**: Can reuse 95% of existing `cluster_run_baseline.py`

**What's the same:**
- ✅ Dataset loading (D1/D2/D3)
- ✅ Configuration system
- ✅ Evaluation framework (all metrics)
- ✅ W&B logging
- ✅ SLURM submission

**What changes:**
- 🔄 Model initialization: `RTDETR()` instead of `YOLO()`
- 🔄 Model weights: `rtdetr-l.pt` instead of `yolov8s.pt`
- 🔄 Argparse choices: Add 'rtdetr' to model options

### **Implementation Plan**

**Option A: Extend existing script (RECOMMENDED)**
```python
# In cluster_run_baseline.py
parser.add_argument('--model', type=str, required=True,
                    choices=['yolov8s', 'yolov11s', 'rtdetr'],  # ← Add rtdetr
                    help='Model architecture to train')

# Model initialization
if config.model_name in ['yolov8s', 'yolov11s', 'yolo11s']:
    model = YOLO(weight_file)
elif config.model_name == 'rtdetr':
    model = RTDETR('rtdetr-l.pt')  # ← Add this
```

**Option B: Separate script**
```
cluster_run_rtdetr.py (new file)
├─ Copy cluster_run_baseline.py
├─ Replace YOLO → RTDETR
└─ Keep everything else identical
```

**Recommendation: Option A** (single unified script)
- Less code duplication
- Easier maintenance
- Same evaluation framework guaranteed

---

## Testing Strategy: Local Before Cluster

### **Local Test Plan**

**Step 1: Test RT-DETR import and loading**
```bash
cd qgfl_experiments
python3 -c "from ultralytics import RTDETR; model = RTDETR('rtdetr-l.pt'); print('RT-DETR ready')"
```

**Step 2: Test on tiny dataset**
```bash
# Use D1 (smallest: 398 images)
# Run for 2 epochs only
python3 cluster_run_baseline.py \
  --model rtdetr \
  --dataset d1 \
  --task binary \
  --epochs 2 \
  --batch-size 4
```

**Expected output:**
- Model loads ✅
- Training starts ✅
- Evaluation runs ✅
- All metrics computed ✅
- W&B logging works ✅

**Step 3: Verify evaluation metrics**
- Check that all metrics are computed (mAP, per-class, confusion matrix, TIDE)
- Verify W&B tables are created
- Confirm same structure as YOLO baseline

**Step 4: Deploy to cluster**
- Upload updated script
- Submit 3 RT-DETR experiments
- Monitor first job closely

---

## Configuration Details

### **RT-DETR Model Variants**

Ultralytics provides several RT-DETR sizes:

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| rtdetr-l | 32M | Medium | High | **RECOMMENDED** |
| rtdetr-x | 76M | Slow | Highest | Optional comparison |

**Recommendation**: Start with `rtdetr-l` (comparable to YOLOv8s/v11s in size)

### **Training Hyperparameters**

**Keep same as YOLO for fair comparison:**
- Epochs: 200
- Batch size: 16
- Image size: 640
- Optimizer: AdamW (RT-DETR default)
- Learning rate: Default (RT-DETR auto-tunes)

### **Dataset Configurations**

**Same as YOLO baselines:**
```yaml
D1 Binary: configs/data_yamls/d1_binary.yaml
D2 Binary: configs/data_yamls/d2_binary.yaml
D3 Binary: configs/data_yamls/d3_binary.yaml
```

**No changes needed** - RT-DETR uses same YOLO format

---

## Hierarchical Tasks: When and How

### **When: After Binary + QGFL (Week 11)**

**Prerequisites before staging/species:**
1. ✅ Binary baselines complete (YOLO + RT-DETR)
2. ✅ QGFL tuned on binary task (optimal hyperparameters found)
3. ✅ Best architecture identified (YOLO vs RT-DETR)
4. ✅ Foundation models explored (RedDino)

### **Staging Task (D1, D2)**

**Dataset preparation:**
```
Current (Binary):
├─ Class 0: Uninfected
└─ Class 1: Infected

Staging (Multi-class):
├─ Class 0: Uninfected
├─ Class 1: Ring
├─ Class 2: Trophozoite
├─ Class 3: Schizont
└─ Class 4: Gametocyte (if present)
```

**Implementation:**
```yaml
# configs/data_yamls/d1_staging.yaml
names:
  0: Uninfected
  1: Ring
  2: Trophozoite
  3: Schizont
  4: Gametocyte

nc: 5  # number of classes
```

**Experiments (Week 11-12):**
```
Staging Baselines:
├─ Best YOLO + D1 staging
├─ Best YOLO + D2 staging
├─ RT-DETR + D1 staging
└─ RT-DETR + D2 staging

Staging + QGFL:
├─ Best model + Best QGFL + D1 staging
└─ Best model + Best QGFL + D2 staging
```

### **Species Task (D3)**

**Dataset preparation:**
```
Species (Multi-class):
├─ Class 0: Uninfected
├─ Class 1: P. falciparum
├─ Class 2: P. vivax
├─ Class 3: P. ovale
└─ Class 4: P. malariae
```

**Implementation:**
```yaml
# configs/data_yamls/d3_species.yaml
names:
  0: Uninfected
  1: P_falciparum
  2: P_vivax
  3: P_ovale
  4: P_malariae

nc: 5  # number of classes
```

**Experiments (Week 13-14):**
```
Species Baselines:
├─ Best YOLO + D3 species
└─ RT-DETR + D3 species

Species + QGFL + Foundation:
├─ RT-DETR + QGFL + D3 species
└─ RedDino + QGFL + D3 species ← Most important
```

### **Why Foundation Models Critical for Species?**

**RedDino advantages for species classification:**
- Pretrained on diverse RBC morphology
- Learned species-specific features
- Transfer learning from 1.25M images
- Better domain generalization

**Expected improvement:**
- Binary task: QGFL alone gives +15-25%
- Species task: RedDino + QGFL gives +30-40% (more complex)

---

## QGFL Implementation Timeline

### **Week 4-5: Standard Focal Loss (Level 1)**
```
Experiments (3):
├─ YOLOv8s + FL + D1 (binary)
├─ YOLOv8s + FL + D2 (binary)
└─ YOLOv8s + FL + D3 (binary)

Config:
- α = 0.9 (infected), 0.1 (uninfected)
- γ = 2.0
```

### **Week 5-6: Difficulty-Aware (Level 2)**
```
Experiments (3):
├─ YOLOv8s + DA + D1
├─ YOLOv8s + DA + D2
└─ YOLOv8s + DA + D3

Config:
- Dynamic γ based on sample difficulty
```

### **Week 6-7: Class-Difficulty (Level 3)**
```
Experiments (3):
├─ YOLOv8s + CD + D1
├─ YOLOv8s + CD + D2
└─ YOLOv8s + CD + D3

Config:
- γ_infected = 8.0 (minority class)
- γ_uninfected = 4.0 (majority class)
```

### **Week 7-8: Complete QGFL (Levels 4-5)**
```
Experiments (6):
├─ YOLOv8s + CD+T + D1/D2/D3 (Level 4)
└─ YOLOv8s + QGFL + D1/D2/D3 (Level 5)

Config:
- Quality-guided weighting
- UIoU integration
- Full framework
```

### **Week 8: RT-DETR + Best QGFL**
```
Experiments (3):
├─ RT-DETR + Best QGFL + D1
├─ RT-DETR + Best QGFL + D2
└─ RT-DETR + Best QGFL + D3

Note: Use best performing QGFL level from YOLO experiments
```

**Total QGFL Experiments: 18** (reduced from 30 by strategic selection)

---

## Ensemble Strategy (Week 15-16)

### **When: After All Individual Models Trained**

**Ensemble components:**
```
Model Pool:
├─ YOLOv8s + QGFL (fast, good minority class)
├─ RT-DETR + QGFL (accurate, global context)
└─ RedDino + QGFL (domain-specific, best features)

Ensemble Methods:
├─ Weighted Box Fusion (WBF)
├─ Non-Maximum Suppression (NMS)
└─ Confidence averaging
```

**Deployment scenarios:**
- **Screening**: Single fast model (YOLO + QGFL)
- **Diagnosis**: Ensemble (all three models)
- **Research**: Best single model (likely RT-DETR + RedDino + QGFL)

---

## Mistake Prevention Checklist

### **Before Each Phase:**

✅ **Code tested locally** (2 epochs, small dataset)
✅ **Evaluation metrics verified** (same as baselines)
✅ **W&B logging confirmed** (tables, metrics, artifacts)
✅ **Configs validated** (correct dataset paths, class names)
✅ **Resource requirements checked** (GPU memory, time estimates)

### **During Experiments:**

✅ **Monitor first job closely** (check logs, metrics)
✅ **Verify W&B updates** (ensure logging works)
✅ **Check intermediate results** (after 50 epochs)
✅ **Compare to baselines** (sanity check performance)

### **After Each Phase:**

✅ **Analyze results thoroughly** (before next phase)
✅ **Document findings** (update analysis notes)
✅ **Adjust hyperparameters** (if needed)
✅ **Plan next experiments** (based on results)

---

## Summary: Clear Experiment Structure

```
Current Status: YOLO Baselines Complete ✅

Next 4 Weeks:
Week 3:   RT-DETR Baselines (3 exp) ← IMMEDIATE NEXT STEP
Week 4-6: QGFL Progressive (15 exp)
Week 7-8: RT-DETR + QGFL (3 exp)
Week 9:   Analysis & Paper Writing

Weeks 9-10: Foundation Models (3 exp)

Weeks 11-14: Hierarchical Tasks ⭐
├─ Week 11-12: Staging (D1, D2)
└─ Week 13-14: Species (D3)

Weeks 15-16: Ensemble & Validation

Total: ~16 weeks (4 months)
```

---

## Decision: RT-DETR Integration Approach

**RECOMMENDED: Extend cluster_run_baseline.py**

**Benefits:**
✅ Single unified script for all architectures
✅ Guaranteed same evaluation framework
✅ Less code duplication
✅ Easier to maintain
✅ Same W&B project/organization

**Next Steps:**
1. Test RT-DETR locally (2 epochs, D1)
2. Add RT-DETR support to cluster_run_baseline.py
3. Test updated script locally
4. Upload to cluster
5. Submit 3 RT-DETR baseline experiments
6. Monitor and validate

**Ready to implement?**
