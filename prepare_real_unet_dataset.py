import os
import glob
import shutil
import random
import numpy as np
from PIL import Image

LGG_PATH = r"C:\Users\G Dineep Chandra\.cache\kagglehub\datasets\mateuszbuda\lgg-mri-segmentation\versions\2\kaggle_3m"
TARGET_DIR = "dataset"

def process_and_save_pair(img_path: str, mask_path: str, dst_img_path: str, dst_mask_path: str):
    # Load RGB/Grayscale image, convert to Grayscale 256x256
    with Image.open(img_path) as img:
        img_gray = img.convert('L').resize((256, 256), Image.Resampling.BILINEAR)
        img_gray.save(dst_img_path, format="PNG")
        
    # Load mask, threshold to binary 0/255 256x256
    with Image.open(mask_path) as mask:
        mask_gray = mask.convert('L').resize((256, 256), Image.Resampling.NEAREST)
        mask_arr = (np.array(mask_gray) > 128).astype(np.uint8) * 255
        Image.fromarray(mask_arr).save(dst_mask_path, format="PNG")

def prepare_unet_dataset(val_ratio: float = 0.15, test_ratio: float = 0.15):
    print("Preparing real LGG Brain Tumor Segmentation UNet Dataset...")
    
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
        
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(TARGET_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_DIR, split, "masks"), exist_ok=True)

    mask_files = sorted(glob.glob(os.path.join(LGG_PATH, "*", "*_mask.tif")))
    pairs = []
    
    for mpath in mask_files:
        ipath = mpath.replace("_mask.tif", ".tif")
        if os.path.exists(ipath):
            pairs.append((ipath, mpath))
            
    # Shuffle deterministically
    random.seed(42)
    random.shuffle(pairs)
    
    total = len(pairs)
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    train_count = total - val_count - test_count
    
    test_pairs = pairs[:test_count]
    val_pairs = pairs[test_count:test_count + val_count]
    train_pairs = pairs[test_count + val_count:]
    
    print(f"Total Real MRI Pairs Found: {total}")
    print(f"  Train pairs : {len(train_pairs)}")
    print(f"  Val pairs   : {len(val_pairs)}")
    print(f"  Test pairs  : {len(test_pairs)} (HELD OUT)")
    
    for split_name, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        for idx, (ipath, mpath) in enumerate(split_pairs):
            dst_img = os.path.join(TARGET_DIR, split_name, "images", f"{split_name}_sample_{idx:04d}.png")
            dst_mask = os.path.join(TARGET_DIR, split_name, "masks", f"{split_name}_sample_{idx:04d}.png")
            process_and_save_pair(ipath, mpath, dst_img, dst_mask)
            
    print("Real UNet Segmentation Dataset Successfully Built!")

if __name__ == "__main__":
    prepare_unet_dataset()
