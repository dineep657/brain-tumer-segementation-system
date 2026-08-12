import os
import numpy as np
from PIL import Image

CLASSES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']

def generate_brisc_structured_dataset():
    """
    Builds the standardized BRISC-aligned 2D dataset structure for both:
    1. Paired Segmentation Task (dataset/train, dataset/val, dataset/test)
    2. 4-Class Classification Task (classification_dataset/train, classification_dataset/val, classification_dataset/test)
    
    Ensures STRICT train/val/test splits with NO overlap between training and testing data.
    """
    dim = 256
    y_grid, x_grid = np.ogrid[:dim, :dim]
    
    brain_center = (128, 128)
    skull_outer_r = 105
    skull_inner_r = 96
    
    outer_mask = (x_grid - brain_center[0])**2 + (y_grid - brain_center[1])**2 <= skull_outer_r**2
    inner_mask = (x_grid - brain_center[0])**2 + (y_grid - brain_center[1])**2 <= skull_inner_r**2
    skull_ring_mask = outer_mask & (~inner_mask)
    
    # 1. Create Directories for Segmentation
    seg_base = "dataset"
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(seg_base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(seg_base, split, "masks"), exist_ok=True)

    # 2. Create Directories for Classification
    cls_base = "classification_dataset"
    for split in ["train", "val", "test"]:
        for cname in CLASSES:
            os.makedirs(os.path.join(cls_base, split, cname), exist_ok=True)

    def create_mri_sample(class_idx: int, seed_id: int):
        np.random.seed(seed_id * 31 + class_idx * 7)
        
        # Base Dark Outer Background (0-15 intensity)
        img_data = np.random.normal(loc=5, scale=3, size=(dim, dim)).astype(np.float32)
        img_data = np.clip(img_data, 0, 15)
        
        # Skull Ring Layer (120-160 intensity)
        skull_tissue = np.random.normal(loc=140, scale=10, size=(dim, dim)).astype(np.float32)
        img_data[skull_ring_mask] = np.clip(skull_tissue[skull_ring_mask], 100, 170)
        
        # Inner Brain Parenchyma (80-140 intensity)
        brain_tissue = np.random.normal(loc=105, scale=12, size=(dim, dim)).astype(np.float32)
        img_data[inner_mask] = np.clip(brain_tissue[inner_mask], 60, 150)
        
        mask_data = np.zeros((dim, dim), dtype=np.uint8)
        cname = CLASSES[class_idx]
        
        if cname == 'glioma':
            # Intra-axial cerebral lesion
            tx, ty, tr = 110, 110, 24
            lesion = ((x_grid - tx)**2 + (y_grid - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=235, scale=8, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 200, 255)
            mask_data[lesion] = 255
            
        elif cname == 'meningioma':
            # Extra-axial dural-based lesion
            tx, ty, tr = 80, 130, 22
            lesion = ((x_grid - tx)**2 + (y_grid - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=225, scale=10, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 190, 250)
            mask_data[lesion] = 255
            
        elif cname == 'pituitary':
            # Sellar/parasellar region lesion
            tx, ty, tr = 128, 165, 20
            lesion = ((x_grid - tx)**2 + (y_grid - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=240, scale=6, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 210, 255)
            mask_data[lesion] = 255
            
        # no_tumor has mask_data = 0
        img_pil = Image.fromarray(img_data.astype(np.uint8))
        mask_pil = Image.fromarray(mask_data)
        return img_pil, mask_pil

    splits = {
        "train": (40, 1),      # 40 per class -> 160 total classification, 192 total segmentation
        "val": (10, 1000),     # 10 per class -> 40 total classification, 48 total segmentation
        "test": (15, 5000)     # 15 per class -> 60 total classification, 60 total segmentation (UNSEEN)
    }

    print("Building BRISC dataset split structure...")
    
    seg_counts = {}
    cls_counts = {}

    for split_name, (num_per_class, seed_offset) in splits.items():
        seg_idx = 0
        for c_idx, cname in enumerate(CLASSES):
            for i in range(num_per_class):
                unique_seed = seed_offset + c_idx * 100 + i
                img, mask = create_mri_sample(c_idx, unique_seed)
                
                # Save to classification split
                cls_img_path = os.path.join(cls_base, split_name, cname, f"{cname}_{i:03d}.png")
                img.save(cls_img_path)
                
                # Save to segmentation split
                fname = f"{split_name}_sample_{seg_idx:03d}.png"
                img.save(os.path.join(seg_base, split_name, "images", fname))
                mask.save(os.path.join(seg_base, split_name, "masks", fname))
                seg_idx += 1
                
        seg_counts[split_name] = seg_idx
        cls_counts[split_name] = len(CLASSES) * num_per_class

    print("Dataset generation complete!")
    print(f"Segmentation Splits Summary: {seg_counts}")
    print(f"Classification Splits Summary: {cls_counts}")

if __name__ == "__main__":
    generate_brisc_structured_dataset()
