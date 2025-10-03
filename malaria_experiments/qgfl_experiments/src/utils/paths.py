# src/utils/paths.py
import os
from pathlib import Path

def get_dataset_paths(dataset_name='d3', task='binary'):
    """Get YOLO-formatted dataset paths"""
    
    # Check environment
    if 'COLAB_GPU' in os.environ:
        base_path = Path("/content/drive/MyDrive/malaria_experiments")
    else:
        base_path = Path("/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments")
    
    # Get or create YOLO format
    from src.utils.coco_to_yolo import get_or_create_yolo_format
    yolo_path = get_or_create_yolo_format(dataset_name, task)
    
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