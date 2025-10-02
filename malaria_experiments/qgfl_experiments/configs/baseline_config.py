# configs/baseline_config.py
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    
    # Dataset
    dataset: str = 'd1'  # 'd1', 'd2', or 'd3'
    task: str = 'binary'  # 'binary', 'species', or 'staging'
    
    # Model
    model_name: str = 'yolov8s'  # or 'yolov11s', 'rtdetr-r18'
    
    # Training
    epochs: int = 200  # Start with 50 for testing
    batch_size: int = 16
    imgsz: int = 640
    device: int = 0 if __import__('torch').cuda.is_available() else 'cpu'
    
    # Optimizer
    optimizer: str = 'SGD'
    lr0: float = 0.005
    momentum: float = 0.95
    weight_decay: float = 0.0005
    
    # Loss (for later)
    use_focal_loss: bool = False
    focal_alpha: float = 0.9
    focal_gamma: float = 2.0
    
    # QGFL parameters (for later)
    use_qgfl: bool = False
    gamma_infected: float = 8.0
    gamma_uninfected: float = 4.0
    
    # Saving
    save_period: int = 10
    patience: int = 20
    
    # Evaluation  
    conf: float = 0.5
    iou: float = 0.5
    
    # W&B
    use_wandb: bool = True
    wandb_key: str = "4024481dae024d54316dd81438ccca923991c6ec"
    wandb_project: str = "malaria_qgfl_experiments"
    
    # Class names
    def get_class_names(self):
        if self.task == 'binary':
            return {0: 'Uninfected', 1: 'Infected'}
        elif self.task == 'species':
            return {
                0: 'Uninfected',
                1: 'P_falciparum', 
                2: 'P_vivax',
                3: 'P_ovale',
                4: 'P_malariae'
            }
        else:  # staging
            if self.dataset == 'd1':
                return {
                    0: 'Uninfected',
                    1: 'Ring',
                    2: 'Trophozoite',
                    3: 'Schizont',
                    4: 'Gametocyte'
                }
            else:  # d2
                return {
                    0: 'Uninfected',
                    1: 'Early',
                    2: 'Intermediate',
                    3: 'Late',
                    4: 'Sexual'
                }
    
    def get_experiment_name(self):
        """Generate experiment name"""
        return f"{self.model_name}_{self.dataset}_{self.task}_baseline"