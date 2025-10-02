# src/utils/coco_to_yolo.py
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import shutil

def coco_to_yolo_bbox(coco_bbox, img_width, img_height):
    """Convert COCO bbox to YOLO format"""
    x, y, w, h = coco_bbox
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def prepare_yolo_structure(dataset_name='d1', task='binary'):
    """
    Prepare YOLO-compatible structure with symlinks
    Creates structure: dataset_dX/yolo_format/{task}/{split}/images & labels
    """
    base_path = Path("/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments")
    dataset_path = base_path / f"dataset_{dataset_name}"
    
    # Create YOLO format directory
    yolo_path = dataset_path / "yolo_format" / task
    
    for split in ['train', 'val', 'test']:
        # Create directories
        split_path = yolo_path / split
        images_path = split_path / "images"
        labels_path = split_path / "labels"
        
        images_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        
        # Find JSON annotation file
        json_candidates = [
            dataset_path / task / split / "_annotations.coco.json",
            dataset_path / task / split / "annotations.coco.json",
            dataset_path / task / split / "_annotations.json",
            dataset_path / task / split / "annotations.json"
        ]
        
        json_path = None
        for candidate in json_candidates:
            if candidate.exists():
                json_path = candidate
                break
        
        if not json_path:
            json_files = list((dataset_path / task / split).glob("*.json"))
            if json_files:
                json_path = json_files[0]
        
        if json_path and json_path.exists():
            print(f"\n=== Processing {dataset_name.upper()}/{task}/{split} ===")
            
            # Load COCO data
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
            
            print(f"Found {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
            
            # Create category mapping
            category_map = {}
            for idx, cat in enumerate(coco_data['categories']):
                category_map[cat['id']] = idx
                print(f"  Class {idx}: {cat['name']}")
            
            # Group annotations by image
            img_annotations = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in img_annotations:
                    img_annotations[img_id] = []
                img_annotations[img_id].append(ann)
            
            # Process each image
            for img_info in tqdm(coco_data['images'], desc=f"Converting {split}"):
                img_id = img_info['id']
                img_filename = img_info['file_name']
                img_width = img_info['width']
                img_height = img_info['height']
                
                # Create symlink for image if not exists
                src_img = dataset_path / "images" / img_filename
                dst_img = images_path / img_filename
                
                if src_img.exists() and not dst_img.exists():
                    dst_img.symlink_to(src_img)
                
                # Create YOLO label file
                txt_filename = Path(img_filename).stem + '.txt'
                txt_path = labels_path / txt_filename
                
                # Write annotations
                if img_id in img_annotations:
                    with open(txt_path, 'w') as f:
                        for ann in img_annotations[img_id]:
                            class_id = category_map[ann['category_id']]
                            x_center, y_center, width, height = coco_to_yolo_bbox(
                                ann['bbox'], img_width, img_height
                            )
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                else:
                    # Empty file for images without annotations
                    txt_path.touch()
            
            print(f"✅ Converted {split}: {len(list(labels_path.glob('*.txt')))} label files")
        else:
            print(f"⚠️  No JSON found for {dataset_name}/{task}/{split}")
    
    return yolo_path

def get_or_create_yolo_format(dataset_name='d1', task='binary'):
    """Get YOLO format path, create if doesn't exist"""
    base_path = Path("/Users/thabangisaka/Downloads/thabang_phd/Experiments/Year 3 Experiments/malaria_experiments")
    yolo_path = base_path / f"dataset_{dataset_name}" / "yolo_format" / task
    
    # Check if already converted
    needs_conversion = False
    for split in ['train', 'val', 'test']:
        labels_path = yolo_path / split / "labels"
        if not labels_path.exists() or len(list(labels_path.glob('*.txt'))) == 0:
            needs_conversion = True
            break
    
    if needs_conversion:
        print(f"Converting {dataset_name}/{task} to YOLO format...")
        yolo_path = prepare_yolo_structure(dataset_name, task)
    else:
        print(f"YOLO format already exists for {dataset_name}/{task}")
    
    return yolo_path