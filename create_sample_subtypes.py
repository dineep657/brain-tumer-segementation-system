import os
import numpy as np
from PIL import Image

def generate_subtype_samples():
    os.makedirs('data', exist_ok=True)
    dim = 256
    y, x = np.ogrid[:dim, :dim]
    
    brain_center = (128, 128)
    brain_radius = 100
    brain_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= brain_radius**2
    
    np.random.seed(42)
    base_brain = np.zeros((dim, dim), dtype=np.float32)
    brain_tissue = np.random.normal(loc=115, scale=14, size=(dim, dim)).astype(np.float32)
    base_brain[brain_mask] = np.clip(brain_tissue[brain_mask], 30, 175)
    
    # 1. Glioma Sample: Irregular infiltrative shape in upper cerebral hemisphere
    glioma_data = base_brain.copy()
    t1 = (x - 150)**2 + (y - 95)**2 <= 28**2
    t2 = (x - 165)**2 + (y - 110)**2 <= 20**2
    glioma_mask = t1 | t2
    glioma_tissue = np.random.normal(loc=235, scale=8, size=(dim, dim)).astype(np.float32)
    glioma_data[glioma_mask] = np.clip(glioma_tissue[glioma_mask], 205, 255)
    Image.fromarray(glioma_data.astype(np.uint8)).save('data/sample_glioma.png')
    print("Glioma sample saved to data/sample_glioma.png")
    
    # 2. Meningioma Sample: Smooth rounded extra-axial mass near skull boundary
    meningioma_data = base_brain.copy()
    meningioma_mask = (x - 85)**2 + (y - 128)**2 <= 24**2
    meningioma_tissue = np.random.normal(loc=240, scale=6, size=(dim, dim)).astype(np.float32)
    meningioma_data[meningioma_mask] = np.clip(meningioma_tissue[meningioma_mask], 215, 255)
    Image.fromarray(meningioma_data.astype(np.uint8)).save('data/sample_meningioma.png')
    print("Meningioma sample saved to data/sample_meningioma.png")
    
    # 3. Pituitary Tumor Sample: Sellar region lower central brain (x=128, y=160)
    pituitary_data = base_brain.copy()
    pituitary_mask = (x - 128)**2 + (y - 160)**2 <= 18**2
    pituitary_tissue = np.random.normal(loc=230, scale=7, size=(dim, dim)).astype(np.float32)
    pituitary_data[pituitary_mask] = np.clip(pituitary_tissue[pituitary_mask], 200, 255)
    Image.fromarray(pituitary_data.astype(np.uint8)).save('data/sample_pituitary.png')
    print("Pituitary tumor sample saved to data/sample_pituitary.png")

if __name__ == "__main__":
    generate_subtype_samples()
