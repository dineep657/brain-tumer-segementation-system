import os
import numpy as np
from PIL import Image

def generate_paired_dataset(num_train: int = 160, num_val: int = 40):
    """
    Generates a paired 2D dataset of synthetic brain MRI images and ground-truth binary segmentation masks.
    Includes explicit dark outer background, skull bone layer, brain parenchyma, and labeled tumor regions.
    """
    base_dir = "dataset"
    train_img_dir = os.path.join(base_dir, "train", "images")
    train_mask_dir = os.path.join(base_dir, "train", "masks")
    val_img_dir = os.path.join(base_dir, "val", "images")
    val_mask_dir = os.path.join(base_dir, "val", "masks")
    
    for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
        os.makedirs(d, exist_ok=True)
        
    dim = 256
    y, x = np.ogrid[:dim, :dim]
    
    brain_center = (128, 128)
    skull_outer_r = 105
    skull_inner_r = 96
    
    outer_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= skull_outer_r**2
    inner_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= skull_inner_r**2
    skull_ring_mask = outer_mask & (~inner_mask)
    
    def create_pair(index: int, is_tumor: bool):
        np.random.seed(index * 17 + (1 if is_tumor else 0))
        
        # 1. Dark outer background (0-15 intensity)
        img_data = np.random.normal(loc=5, scale=3, size=(dim, dim)).astype(np.float32)
        img_data = np.clip(img_data, 0, 15)
        
        # 2. Skull ring layer (medium-high intensity 120-160)
        skull_tissue = np.random.normal(loc=140, scale=10, size=(dim, dim)).astype(np.float32)
        img_data[skull_ring_mask] = np.clip(skull_tissue[skull_ring_mask], 100, 170)
        
        # 3. Inner brain tissue (80-140 intensity)
        brain_tissue = np.random.normal(loc=105, scale=12, size=(dim, dim)).astype(np.float32)
        img_data[inner_mask] = np.clip(brain_tissue[inner_mask], 60, 150)
        
        # Ground-truth binary mask (0 background/skull/healthy tissue)
        mask_data = np.zeros((dim, dim), dtype=np.uint8)
        
        if is_tumor:
            # Generate tumor lesion inside inner brain tissue
            tx = np.random.randint(95, 161)
            ty = np.random.randint(95, 161)
            tr = np.random.randint(14, 26)
            
            tumor_region = ((x - tx)**2 + (y - ty)**2 <= tr**2) & inner_mask
            tumor_tissue = np.random.normal(loc=235, scale=8, size=(dim, dim)).astype(np.float32)
            
            img_data[tumor_region] = np.clip(tumor_tissue[tumor_region], 200, 255)
            mask_data[tumor_region] = 255
            
        img_pil = Image.fromarray(img_data.astype(np.uint8))
        mask_pil = Image.fromarray(mask_data)
        return img_pil, mask_pil

    print(f"Generating {num_train} paired training samples...")
    for i in range(num_train):
        is_tumor = (i % 2 == 0)
        img, mask = create_pair(i, is_tumor)
        img.save(os.path.join(train_img_dir, f"sample_{i:03d}.png"))
        mask.save(os.path.join(train_mask_dir, f"sample_{i:03d}.png"))
        
    print(f"Generating {num_val} paired validation samples...")
    for i in range(num_val):
        is_tumor = (i % 2 == 0)
        img, mask = create_pair(2000 + i, is_tumor)
        img.save(os.path.join(val_img_dir, f"val_sample_{i:03d}.png"))
        mask.save(os.path.join(val_mask_dir, f"val_sample_{i:03d}.png"))
        
    print(f"Paired anatomical dataset generated successfully in '{base_dir}/'.")

if __name__ == "__main__":
    generate_paired_dataset()
