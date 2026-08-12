import os
import numpy as np
from PIL import Image

def generate_sample_2d_mri():
    os.makedirs('data', exist_ok=True)
    dim = 256
    
    # 2D Grid
    y, x = np.ogrid[:dim, :dim]
    
    # Background brain ellipse/circle
    brain_center = (128, 128)
    brain_radius = 100
    brain_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= brain_radius**2
    
    # Base image array
    data = np.zeros((dim, dim), dtype=np.float32)
    
    # Brain tissue background with noise
    np.random.seed(42)
    brain_tissue = np.random.normal(loc=120, scale=15, size=(dim, dim)).astype(np.float32)
    data[brain_mask] = np.clip(brain_tissue[brain_mask], 30, 180)
    
    # Tumor region (center 145, 110, radius 30)
    tumor_center = (145, 110)
    tumor_radius = 30
    tumor_mask = (x - tumor_center[0])**2 + (y - tumor_center[1])**2 <= tumor_radius**2
    
    # High-intensity tumor tissue with noise
    tumor_tissue = np.random.normal(loc=230, scale=10, size=(dim, dim)).astype(np.float32)
    data[tumor_mask] = np.clip(tumor_tissue[tumor_mask], 200, 255)
    
    # Convert to uint8 grayscale image
    img_uint8 = data.astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    output_path = os.path.join('data', 'sample_2d_mri.png')
    img.save(output_path)
    print(f"Synthetic 2D MRI slice saved to: {output_path} (Dimensions: {img.size})")

if __name__ == "__main__":
    generate_sample_2d_mri()
