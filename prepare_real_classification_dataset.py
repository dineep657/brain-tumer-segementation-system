import os
import shutil
import random
from PIL import Image
from typing import Tuple

KAGGLE_PATH = r"C:\Users\G Dineep Chandra\.cache\kagglehub\datasets\masoudnickparvar\brain-tumor-mri-dataset\versions\2"
TARGET_DIR = "classification_dataset"

CLASS_MAPPING = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary",
    "notumor": "No Tumor"
}

def process_and_save_image(src_path: str, dst_path: str, size: Tuple[int, int] = (256, 256)):
    """Preprocesses a real MRI image into 256x256 grayscale PNG."""
    with Image.open(src_path) as img:
        img_gray = img.convert('L').resize(size, Image.Resampling.BILINEAR)
        img_gray.save(dst_path, format="PNG")

def prepare_dataset(val_ratio: float = 0.20):
    print("Preparing real Kaggle Brain Tumor MRI Classification Dataset...")
    
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        
    for split in ["train", "val", "test"]:
        for target_class in CLASS_MAPPING.values():
            os.makedirs(os.path.join(TARGET_DIR, split, target_class), exist_ok=True)

    # 1. Process Training & Validation Splits
    train_src_dir = os.path.join(KAGGLE_PATH, "Training")
    counts = {"train": 0, "val": 0, "test": 0}
    
    for raw_class, target_class in CLASS_MAPPING.items():
        class_folder = os.path.join(train_src_dir, raw_class)
        files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        random.seed(42)
        random.shuffle(files)
        
        val_count = int(len(files) * val_ratio)
        val_files = files[:val_count]
        train_files = files[val_count:]
        
        for idx, fname in enumerate(train_files):
            src = os.path.join(class_folder, fname)
            dst = os.path.join(TARGET_DIR, "train", target_class, f"train_{raw_class}_{idx:04d}.png")
            process_and_save_image(src, dst)
            counts["train"] += 1
            
        for idx, fname in enumerate(val_files):
            src = os.path.join(class_folder, fname)
            dst = os.path.join(TARGET_DIR, "val", target_class, f"val_{raw_class}_{idx:04d}.png")
            process_and_save_image(src, dst)
            counts["val"] += 1

    # 2. Process Held-Out Test Split
    test_src_dir = os.path.join(KAGGLE_PATH, "Testing")
    for raw_class, target_class in CLASS_MAPPING.items():
        class_folder = os.path.join(test_src_dir, raw_class)
        files = sorted([f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for idx, fname in enumerate(files):
            src = os.path.join(class_folder, fname)
            dst = os.path.join(TARGET_DIR, "test", target_class, f"test_{raw_class}_{idx:04d}.png")
            process_and_save_image(src, dst)
            counts["test"] += 1

    print("Real Classification Dataset Successfully Built!")
    print(f"  Training Set   : {counts['train']} images (~1120 per class)")
    print(f"  Validation Set : {counts['val']} images (~280 per class)")
    print(f"  Test Set       : {counts['test']} images (400 per class - HELD OUT)")

if __name__ == "__main__":
    prepare_dataset()
