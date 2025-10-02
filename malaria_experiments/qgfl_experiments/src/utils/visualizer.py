# src/utils/visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from pathlib import Path
import random

class YOLOVisualizer:
    """Modular visualizer that adapts to any task"""
    
    def __init__(self, class_names):
        self.class_names = class_names
        self.colors = self._generate_colors(class_names)
        
        # Auto-detect minority classes (for prioritization)
        self.minority_classes = self._detect_minority_classes(class_names)
    
    def _detect_minority_classes(self, class_names):
        """Auto-detect which classes are likely minority (e.g., infected, disease stages)"""
        minority_keywords = ['infected', 'falciparum', 'vivax', 'ovale', 'malariae', 
                           'ring', 'trophozoite', 'schizont', 'gametocyte', 
                           'early', 'intermediate', 'late', 'sexual']
        minority = []
        for class_id, name in class_names.items():
            name_lower = name.lower()
            # Check if it's a minority class (not uninfected/normal)
            if any(keyword in name_lower for keyword in minority_keywords) and 'uninfected' not in name_lower:
                minority.append(class_id)
        return minority
    
    def _generate_colors(self, class_names):
        """Generate distinct colors based on number of classes"""
        n_classes = len(class_names)
        colors = []
        
        if n_classes == 2:
            # Binary: Green for majority, Red for minority
            for i in range(n_classes):
                name = class_names.get(i, '').lower()
                if 'uninfected' in name or 'normal' in name:
                    colors.append([0.0, 0.8, 0.0, 1.0])  # Green
                else:
                    colors.append([1.0, 0.0, 0.0, 1.0])  # Red
        else:
            # Multi-class: Use distinct colors
            cmap = plt.cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
            for i in range(n_classes):
                colors.append(cmap(i % cmap.N)[:4])
        
        return colors
    
    def load_yolo_annotations(self, img_path, label_path):
        """Load image and its YOLO annotations"""
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * img_width
                        y_center = float(parts[2]) * img_height
                        width = float(parts[3]) * img_width
                        height = float(parts[4]) * img_height
                        
                        x1 = x_center - width/2
                        y1 = y_center - height/2
                        
                        boxes.append({
                            'class_id': class_id,
                            'x': x1,
                            'y': y1,
                            'width': width,
                            'height': height,
                            'class_name': self.class_names.get(class_id, f"Class_{class_id}"),
                            'is_minority': class_id in self.minority_classes
                        })
        
        return img, boxes
    
    def plot_image_with_boxes(self, ax, img, boxes, title=""):
        """Plot single image with bounding boxes"""
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Count boxes by class
        class_counts = {}
        for box in boxes:
            class_counts[box['class_id']] = class_counts.get(box['class_id'], 0) + 1
        
        # Draw boxes
        for box in boxes:
            # Thicker lines for minority classes
            linewidth = 2.5 if box['is_minority'] else 1.5
            
            rect = patches.Rectangle(
                (box['x'], box['y']),
                box['width'],
                box['height'],
                linewidth=linewidth,
                edgecolor=self.colors[box['class_id']],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Show labels for minority classes or if few boxes
            if box['is_minority'] or len(boxes) < 20:
                ax.text(
                    box['x'], 
                    box['y'] - 5,
                    f"{box['class_name']}",
                    color='white',
                    fontsize=7,
                    fontweight='bold' if box['is_minority'] else 'normal',
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor=self.colors[box['class_id']],
                        alpha=0.9
                    )
                )
        
        # Add summary statistics
        stats_text = f"Total: {len(boxes)}\n"
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            class_name = self.class_names.get(class_id, f"Class_{class_id}")
            # Shorten long names
            if len(class_name) > 12:
                class_name = class_name[:10] + ".."
            stats_text += f"{class_name}: {count}\n"
        
        ax.text(
            0.02, 0.98, 
            stats_text.strip(),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    def visualize_dataset_samples(self, yolo_path, num_samples=3):
        """Visualize samples from train, val, and test splits"""
        splits = ['train', 'val', 'test']
        
        fig, axes = plt.subplots(len(splits), num_samples, figsize=(5*num_samples, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)
        
        # Create informative title
        n_classes = len(self.class_names)
        task_type = "Binary" if n_classes == 2 else f"{n_classes}-Class"
        
        fig.suptitle(f'Dataset Verification ({task_type} Classification)\n' + 
                    f'Classes: {", ".join(self.class_names.values())}', 
                    fontsize=12, fontweight='bold')
        
        for split_idx, split in enumerate(splits):
            img_dir = yolo_path / split / "images"
            lbl_dir = yolo_path / split / "labels"
            
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            
            if len(img_files) == 0:
                for j in range(num_samples):
                    axes[split_idx, j].text(0.5, 0.5, f"No images in {split}", 
                                           ha='center', va='center')
                    axes[split_idx, j].axis('off')
                continue
            
            # Prioritize images with minority classes if they exist
            if self.minority_classes:
                minority_images = []
                for img_path in img_files:
                    label_path = lbl_dir / (img_path.stem + '.txt')
                    if label_path.exists():
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5 and int(parts[0]) in self.minority_classes:
                                    minority_images.append(img_path)
                                    break
                
                if minority_images:
                    sample_files = random.sample(minority_images, min(num_samples, len(minority_images)))
                else:
                    sample_files = random.sample(img_files, min(num_samples, len(img_files)))
            else:
                sample_files = random.sample(img_files, min(num_samples, len(img_files)))
            
            for j, img_path in enumerate(sample_files):
                label_path = lbl_dir / (img_path.stem + '.txt')
                img, boxes = self.load_yolo_annotations(img_path, label_path)
                
                title = f"{split.upper()}: {img_path.name[:20]}..."
                self.plot_image_with_boxes(axes[split_idx, j], img, boxes, title)
        
        plt.tight_layout()
        return fig
    
    def get_dataset_stats(self, yolo_path):
        """Get comprehensive statistics about the dataset"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            img_dir = yolo_path / split / "images"
            lbl_dir = yolo_path / split / "labels"
            
            if not img_dir.exists():
                continue
            
            img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
            lbl_files = list(lbl_dir.glob('*.txt'))
            
            class_counts = {i: 0 for i in range(len(self.class_names))}
            total_boxes = 0
            empty_labels = 0
            images_with_minority = 0
            
            for lbl_file in lbl_files:
                classes_in_image = set()
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        empty_labels += 1
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            classes_in_image.add(class_id)
                            total_boxes += 1
                
                if any(c in self.minority_classes for c in classes_in_image):
                    images_with_minority += 1
            
            stats[split] = {
                'images': len(img_files),
                'labels': len(lbl_files),
                'total_boxes': total_boxes,
                'empty_labels': empty_labels,
                'images_with_minority': images_with_minority,
                'class_distribution': class_counts,
                'avg_boxes_per_image': total_boxes / len(img_files) if img_files else 0
            }
        
        return stats