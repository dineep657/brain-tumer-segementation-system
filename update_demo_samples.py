import os
import shutil
from PIL import Image

def update_samples():
    print("Updating demo sample images in data/ with real MRI test set crops...")
    os.makedirs("data", exist_ok=True)
    
    test_dir = os.path.join("classification_dataset", "test")
    
    mapping = {
        "Glioma": "sample_glioma.png",
        "Meningioma": "sample_meningioma.png",
        "Pituitary": "sample_pituitary.png",
        "No Tumor": "sample_normal_mri.png"
    }
    
    for cname, target_fname in mapping.items():
        cdir = os.path.join(test_dir, cname)
        if os.path.exists(cdir):
            files = sorted([f for f in os.listdir(cdir) if f.endswith('.png')])
            if files:
                src = os.path.join(cdir, files[0])
                dst = os.path.join("data", target_fname)
                shutil.copy(src, dst)
                print(f"  Copied real {cname} test scan -> '{dst}'")

if __name__ == "__main__":
    update_samples()
