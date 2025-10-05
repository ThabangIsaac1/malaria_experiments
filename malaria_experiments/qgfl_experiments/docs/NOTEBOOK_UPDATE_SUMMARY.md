# Notebook Update Summary - Per-Class mAP@50-95 Addition

## Date: 2025-10-05

## Changes Required for `01_run_baseline.ipynb`

### 1. W&B Per-Class Charts (Cell ~311)

**BEFORE:**
```python
# 1.4 Per-Class Performance Charts - Validation
wandb.log({
    f'{prefix}/mAP50': metrics.get('mAP50', 0),  # ← REMOVE (doesn't exist)
    f'{prefix}/precision': metrics.get('precision', 0),
    f'{prefix}/recall': metrics.get('recall', 0),
    f'{prefix}/f1_score': metrics.get('f1_score', 0),
})

# 1.5 Per-Class Performance Charts - Test
wandb.log({
    f'{prefix}/mAP50': metrics.get('mAP50', 0),  # ← REMOVE (doesn't exist)
    f'{prefix}/precision': metrics.get('precision', 0),
    f'{prefix}/recall': metrics.get('recall', 0),
    f'{prefix}/f1_score': metrics.get('f1_score', 0),
})
```

**AFTER:**
```python
# 1.4 Per-Class Performance Charts - Validation
wandb.log({
    f'{prefix}/precision': metrics.get('precision', 0),
    f'{prefix}/recall': metrics.get('recall', 0),
    f'{prefix}/f1_score': metrics.get('f1_score', 0),
    f'{prefix}/mAP50_95': float(val_results.get('global', {}).get('per_class_map', {}).get(class_name, 0)),  # ← ADD
})

# 1.5 Per-Class Performance Charts - Test
wandb.log({
    f'{prefix}/precision': metrics.get('precision', 0),
    f'{prefix}/recall': metrics.get('recall', 0),
    f'{prefix}/f1_score': metrics.get('f1_score', 0),
    f'{prefix}/mAP50_95': float(test_results.get('global', {}).get('per_class_map', {}).get(class_name, 0)),  # ← ADD
})
```

### 2. W&B Validation Per-Class Table (Cell ~312)

**BEFORE:**
```python
val_class_data.append({
    'Class': class_name,
    # 'mAP50': float(m.get('mAP50', 0)),  # FORCE FLOAT
    'Precision': float(m.get('precision', 0)),
    'Recall': float(m.get('recall', 0)),
    'F1': float(m.get('f1_score', 0)),
    'Support': int(m.get('support', 0))
})
```

**AFTER:**
```python
val_class_data.append({
    'Class': class_name,
    'Precision': float(m.get('precision', 0)),
    'Recall': float(m.get('recall', 0)),
    'F1': float(m.get('f1_score', 0)),
    'mAP50-95': float(val_results.get('global', {}).get('per_class_map', {}).get(class_name, 0)),  # ← ADD
    'Support': int(m.get('support', 0))
})
```

### 3. W&B Test Per-Class Table (Cell ~312)

**BEFORE:**
```python
test_class_data.append({
    'Class': class_name,
    # 'mAP50': float(m.get('mAP50', 0)),  # FORCE FLOAT
    'Precision': float(m.get('precision', 0)),
    'Recall': float(m.get('recall', 0)),
    'F1': float(m.get('f1_score', 0)),
    'AP': float(test_results.get('pr_analysis', {}).get(class_name, {}).get('ap', 0)),
    'Support': int(m.get('support', 0)),
    'TP': int(m.get('tp', 0)),
    'FP': int(m.get('fp', 0)),
    'FN': int(m.get('fn', 0))
})
```

**AFTER:**
```python
test_class_data.append({
    'Class': class_name,
    'Precision': float(m.get('precision', 0)),
    'Recall': float(m.get('recall', 0)),
    'F1': float(m.get('f1_score', 0)),
    'AP': float(test_results.get('pr_analysis', {}).get(class_name, {}).get('ap', 0)),
    'mAP50-95': float(test_results.get('global', {}).get('per_class_map', {}).get(class_name, 0)),  # ← ADD
    'Support': int(m.get('support', 0)),
    'TP': int(m.get('tp', 0)),
    'FP': int(m.get('fp', 0)),
    'FN': int(m.get('fn', 0))
})
```

### 4. W&B Precision-Recall Analysis Table (Cell ~312)

**BEFORE:**
```python
pr_data.append({
    'Class': class_name,
    'AP': pr_stats.get('ap', 0),
    'Optimal_Threshold': pr_stats.get('optimal_threshold', 0),
    'Precision_at_Optimal': pr_stats.get('precision_at_optimal', 0),
    'Recall_at_Optimal': pr_stats.get('recall_at_optimal', 0),
    'Max_F1': pr_stats.get('max_f1', 0)
})
```

**AFTER:**
```python
pr_data.append({
    'Class': class_name,
    'AP': pr_stats.get('ap', 0),
    'mAP50-95': float(test_results.get('global', {}).get('per_class_map', {}).get(class_name, 0)),  # ← ADD
    'Optimal_Threshold': pr_stats.get('optimal_threshold', 0),
    'Precision_at_Optimal': pr_stats.get('precision_at_optimal', 0),
    'Recall_at_Optimal': pr_stats.get('recall_at_optimal', 0),
    'Max_F1': pr_stats.get('max_f1', 0)
})
```

## Summary

**Total Changes:** 5 code blocks across 2 cells
- **Cell ~311:** Per-class charts logging (2 changes)
- **Cell ~312:** Per-class table creation (3 changes)

**What This Adds:**
- Per-class mAP@50-95 to W&B time-series charts
- Per-class mAP@50-95 to validation_per_class table
- Per-class mAP@50-95 to test_per_class table
- Per-class mAP@50-95 to precision_recall_analysis table

**Source of Data:**
- Extracted from YOLO's `metrics.box.maps` in evaluator.py
- Stored in `results['global']['per_class_map']`
- Maps class names to their mAP@50-95 values

## Verification

After updating, the notebook should produce tables with:
- **Validation table:** 6 columns (Class, Precision, Recall, F1, mAP50-95, Support)
- **Test table:** 10 columns (Class, Precision, Recall, F1, AP, mAP50-95, Support, TP, FP, FN)
- **PR analysis:** 7 columns (Class, AP, mAP50-95, Optimal_Threshold, Precision_at_Optimal, Recall_at_Optimal, Max_F1)
