import os
import numpy as np
from PIL import Image

CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

def generate_classification_dataset(num_per_class_train: int = 40, num_per_class_val: int = 10):
    """
    Generates a 4-class labeled dataset of 2D brain MRI images for training a PyTorch multi-class classifier.
    Classes: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
    """
    base_dir = "classification_dataset"
    dim = 256
    y, x = np.ogrid[:dim, :dim]
    
    brain_center = (128, 128)
    skull_outer_r = 105
    skull_inner_r = 96
    
    outer_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= skull_outer_r**2
    inner_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= skull_inner_r**2
    skull_ring_mask = outer_mask & (~inner_mask)
    
    for split in ["train", "val"]:
        for cname in CLASSES:
            os.makedirs(os.path.join(base_dir, split, cname), exist_ok=True)

    def create_sample(class_idx: int, sample_idx: int):
        np.random.seed(class_idx * 1000 + sample_idx * 17)
        
        # 1. Background (0-15 intensity)
        img_data = np.random.normal(loc=5, scale=3, size=(dim, dim)).astype(np.float32)
        img_data = np.clip(img_data, 0, 15)
        
        # 2. Skull ring (120-160 intensity)
        skull_tissue = np.random.normal(loc=140, scale=10, size=(dim, dim)).astype(np.float32)
        img_data[skull_ring_mask] = np.clip(skull_tissue[skull_ring_mask], 100, 170)
        
        # 3. Inner brain tissue (80-140 intensity)
        brain_tissue = np.random.normal(loc=105, scale=12, size=(dim, dim)).astype(np.float32)
        img_data[inner_mask] = np.clip(brain_tissue[inner_mask], 60, 150)
        
        cname = CLASSES[class_idx]
        
        if cname == 'Glioma':
            # Intra-axial cerebral lesion (center region)
            tx, ty, tr = 110, 110, 24
            lesion = ((x - tx)**2 + (y - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=235, scale=8, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 200, 255)
            
        elif cname == 'Meningioma':
            # Extra-axial dural-based lesion (peripheral region)
            tx, ty, tr = 80, 130, 22
            lesion = ((x - tx)**2 + (y - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=225, scale=10, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 190, 250)
            
        elif cname == 'Pituitary':
            # Sellar/parasellar region lesion (lower central region)
            tx, ty, tr = 128, 165, 20
            lesion = ((x - tx)**2 + (y - ty)**2 <= tr**2) & inner_mask
            lesion_intensity = np.random.normal(loc=240, scale=6, size=(dim, dim)).astype(np.float32)
            img_data[lesion] = np.clip(lesion_intensity[lesion], 210, 255)
            
        # No Tumor has zero lesion added
        return Image.fromarray(img_data.astype(np.uint8))

    print("Generating 4-class labeled training samples...")
    for c_idx, cname in enumerate(CLASSES):
        for i in range(num_per_class_train):
            img = create_sample(c_idx, i)
            img.save(os.path.join(base_dir, "train", cname, f"{cname.lower()}_{i:03d}.png"))

    print("Generating 4-class labeled validation samples...")
    for c_idx, cname in enumerate(CLASSES):
        for i in range(num_per_class_val):
            img = create_sample(c_idx, 500 + i)
            img.save(os.path.join(base_dir, "val", cname, f"{cname.lower()}_val_{i:03d}.png"))

    print(f"Classification dataset generated successfully in '{base_dir}/'.")

if __name__ == "__main__":
    generate_classification_dataset()
