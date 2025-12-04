"""
Configuration for Dynamic Adaptive QGFL

Provides easy configuration for different tasks (binary/staging/species),
models (YOLO/RT-DETR), and datasets (D1/D2/D3).

Author: Thabang Isaka
Date: 2025-11-21
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class AdaptiveQGFLConfig:
    """
    Configuration for Dynamic Adaptive QGFL

    Designed for easy transitions between:
    - Tasks: binary, staging, species
    - Models: YOLO, RT-DETR
    - Datasets: D1, D2, D3

    Auto-configures nc, class_names, and initial parameters based on task/dataset.
    """

    # === TASK CONFIGURATION ===
    task: str = 'binary'  # 'binary', 'staging', 'species'
    nc: int = 2  # Auto-set based on task
    class_names: Dict[int, str] = field(default_factory=dict)

    # === MODEL CONFIGURATION ===
    model_type: str = 'yolo'  # 'yolo', 'rtdetr'

    # === DATASET CONFIGURATION ===
    dataset: str = 'd1'  # 'd1', 'd2', 'd3'

    # === INITIAL PARAMETERS ===
    # Starting points - adaptation takes over during training
    initial_alpha: float = 0.75
    initial_gamma: float = 6.0
    initial_threshold: float = 0.5

    # === ADAPTATION SETTINGS ===
    adaptation_rate: float = 0.05  # How fast to adapt (0.01-0.1)
    adaptation_momentum: float = 0.9  # Smoothing factor (0.8-0.95)
    warmup_epochs: int = 5  # Don't adapt during warmup
    freeze_final_epochs: int = 20  # Freeze adaptation in final N epochs

    # === TARGET METRICS ===
    target_recall: float = 0.90  # Target recall for minority classes
    min_precision: float = 0.50  # Minimum acceptable precision

    # === PARAMETER BOUNDS ===
    alpha_bounds: Tuple[float, float] = (0.3, 0.95)
    gamma_bounds: Tuple[float, float] = (2.0, 10.0)
    threshold_bounds: Tuple[float, float] = (0.3, 0.95)

    # === INHERITED QGFL PARAMS ===
    quality_margin: float = 0.5
    quality_factor: float = 2.0
    uiou_start: float = 2.0
    uiou_end: float = 0.5

    # === LOGGING ===
    use_wandb: bool = True
    debug: bool = True

    def __post_init__(self):
        """Auto-configure based on task and dataset"""
        self._configure_task()
        self._apply_dataset_hints()

    def _configure_task(self):
        """Set nc and class_names based on task"""
        task_configs = {
            'binary': {
                'nc': 2,
                'class_names': {0: 'Uninfected', 1: 'Infected'}
            },
            'staging': {
                'nc': 5,
                'class_names': {
                    0: 'Uninfected',
                    1: 'Ring',
                    2: 'Trophozoite',
                    3: 'Schizont',
                    4: 'Gametocyte'
                }
            },
            'species': {
                'nc': 5,
                'class_names': {
                    0: 'Uninfected',
                    1: 'P_falciparum',
                    2: 'P_vivax',
                    3: 'P_ovale',
                    4: 'P_malariae'
                }
            }
        }

        if self.task in task_configs:
            config = task_configs[self.task]
            self.nc = config['nc']
            if not self.class_names:
                self.class_names = config['class_names']
        else:
            raise ValueError(f"Unknown task: {self.task}. Choose from: {list(task_configs.keys())}")

    def _apply_dataset_hints(self):
        """
        Apply dataset-specific initial parameter hints.
        Based on experimental findings:
        - D1: Strong baseline (R=0.95) needs alpha=0.65
        - D2: Weak baseline (R=0.43) needs alpha=0.85
        - D3: Strong baseline (R=0.92) needs alpha=0.65
        """
        dataset_hints = {
            'd1': {'initial_alpha': 0.65, 'initial_gamma': 5.0},
            'd2': {'initial_alpha': 0.85, 'initial_gamma': 7.0},
            'd3': {'initial_alpha': 0.65, 'initial_gamma': 5.0},
        }

        if self.dataset in dataset_hints:
            hints = dataset_hints[self.dataset]
            # Only apply hints if user didn't override defaults
            if self.initial_alpha == 0.75:
                self.initial_alpha = hints['initial_alpha']
            if self.initial_gamma == 6.0:
                self.initial_gamma = hints['initial_gamma']

    def get_minority_class_ids(self) -> list:
        """Return list of minority class IDs (all except class 0)"""
        return list(range(1, self.nc))

    def get_majority_class_id(self) -> int:
        """Return majority class ID (always 0 for Uninfected)"""
        return 0

    def to_dict(self) -> Dict:
        """Convert config to dictionary for W&B logging"""
        return {
            'task': self.task,
            'nc': self.nc,
            'class_names': self.class_names,
            'model_type': self.model_type,
            'dataset': self.dataset,
            'initial_alpha': self.initial_alpha,
            'initial_gamma': self.initial_gamma,
            'initial_threshold': self.initial_threshold,
            'adaptation_rate': self.adaptation_rate,
            'adaptation_momentum': self.adaptation_momentum,
            'warmup_epochs': self.warmup_epochs,
            'target_recall': self.target_recall,
            'min_precision': self.min_precision,
            'alpha_bounds': self.alpha_bounds,
            'gamma_bounds': self.gamma_bounds,
        }


# ============================================================================
# UNIT TESTS
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing AdaptiveQGFLConfig")
    print("=" * 60)

    # Test 1: Binary task configuration
    print("\nTest 1: Binary task configuration")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1')
    assert config.nc == 2, f"Expected nc=2, got {config.nc}"
    assert config.class_names == {0: 'Uninfected', 1: 'Infected'}
    assert config.initial_alpha == 0.65, f"D1 should hint alpha=0.65, got {config.initial_alpha}"
    print(f"  nc={config.nc}, class_names={config.class_names}")
    print(f"  initial_alpha={config.initial_alpha} (D1 hint applied)")
    print("  PASSED")

    # Test 2: Staging task configuration
    print("\nTest 2: Staging task configuration")
    config = AdaptiveQGFLConfig(task='staging', dataset='d1')
    assert config.nc == 5, f"Expected nc=5, got {config.nc}"
    assert len(config.class_names) == 5
    assert 'Schizont' in config.class_names.values()
    print(f"  nc={config.nc}")
    print(f"  classes: {list(config.class_names.values())}")
    print("  PASSED")

    # Test 3: Species task configuration
    print("\nTest 3: Species task configuration")
    config = AdaptiveQGFLConfig(task='species', dataset='d3')
    assert config.nc == 5, f"Expected nc=5, got {config.nc}"
    assert 'P_falciparum' in config.class_names.values()
    print(f"  nc={config.nc}")
    print(f"  classes: {list(config.class_names.values())}")
    print("  PASSED")

    # Test 4: D2 dataset hint (weak baseline)
    print("\nTest 4: D2 dataset hint (weak baseline)")
    config = AdaptiveQGFLConfig(task='binary', dataset='d2')
    assert config.initial_alpha == 0.85, f"D2 should hint alpha=0.85, got {config.initial_alpha}"
    print(f"  initial_alpha={config.initial_alpha} (D2 weak baseline hint)")
    print("  PASSED")

    # Test 5: User override
    print("\nTest 5: User override of initial parameters")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1', initial_alpha=0.80)
    assert config.initial_alpha == 0.80, f"User override should take precedence"
    print(f"  initial_alpha={config.initial_alpha} (user override)")
    print("  PASSED")

    # Test 6: Minority class IDs
    print("\nTest 6: Minority class ID helpers")
    config = AdaptiveQGFLConfig(task='staging')
    minority = config.get_minority_class_ids()
    majority = config.get_majority_class_id()
    assert minority == [1, 2, 3, 4], f"Expected [1,2,3,4], got {minority}"
    assert majority == 0
    print(f"  minority_class_ids={minority}")
    print(f"  majority_class_id={majority}")
    print("  PASSED")

    # Test 7: to_dict for W&B
    print("\nTest 7: Config to dictionary")
    config = AdaptiveQGFLConfig(task='binary', dataset='d1')
    config_dict = config.to_dict()
    assert 'task' in config_dict
    assert 'initial_alpha' in config_dict
    print(f"  Keys: {list(config_dict.keys())[:5]}...")
    print("  PASSED")

    print("\n" + "=" * 60)
    print("ALL CONFIG TESTS PASSED")
    print("=" * 60)
