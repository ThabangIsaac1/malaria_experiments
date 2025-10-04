# src/utils/paths.py
import os
from pathlib import Path

def get_dataset_paths(dataset_name='d3', task='binary'):
    """Get YOLO-formatted dataset paths"""

    # Auto-detect base path based on environment
    if 'COLAB_GPU' in os.environ:
        # Google Colab
        base_path = Path("/content/drive/MyDrive/malaria_experiments")
    elif Path.cwd().name == 'qgfl_experiments':
        # Running from qgfl_experiments directory
        base_path = Path.cwd().parent
    elif (Path.cwd() / 'qgfl_experiments').exists():
        # Running from malaria_experiments directory
        base_path = Path.cwd()
    elif (Path.cwd().parent / 'malaria_experiments').exists():
        # Running from subdirectory
        base_path = Path.cwd().parent / 'malaria_experiments'
    else:
        # Fallback: search upward for malaria_experiments
        current = Path.cwd()
        while current != current.parent:
            if (current / 'dataset_d1').exists() or current.name == 'malaria_experiments':
                base_path = current
                break
            current = current.parent
        else:
            raise FileNotFoundError("Cannot find malaria_experiments base directory. Please run from project directory.")

    # Get or create YOLO format
    from src.utils.coco_to_yolo import get_or_create_yolo_format
    yolo_path = get_or_create_yolo_format(dataset_name, task, base_path)

    return yolo_path

def verify_dataset(dataset_name='d3', task='binary'):
    """Verify YOLO dataset structure"""
    yolo_path = get_dataset_paths(dataset_name, task)
    
    print(f"\nDataset: {dataset_name.upper()} - Task: {task}")
    print(f"YOLO Path: {yolo_path}")
    print("-" * 50)
    
    total_images = 0
    total_labels = 0
    
    for split in ['train', 'val', 'test']:
        img_path = yolo_path / split / "images"
        lbl_path = yolo_path / split / "labels"
        
        if img_path.exists():
            img_count = len(list(img_path.glob('*.jpg')) + list(img_path.glob('*.png')))
            total_images += img_count
            print(f"✅ {split}/images: {img_count} files")
        else:
            print(f"❌ {split}/images: NOT FOUND")
            
        if lbl_path.exists():
            lbl_count = len(list(lbl_path.glob('*.txt')))
            total_labels += lbl_count
            print(f"✅ {split}/labels: {lbl_count} files")
        else:
            print(f"❌ {split}/labels: NOT FOUND")
    
    print(f"\nTotal: {total_images} images, {total_labels} labels")
    return total_images > 0