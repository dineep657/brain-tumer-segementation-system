import os
import numpy as np
from PIL import Image

def generate_sample_normal_2d():
    os.makedirs('data', exist_ok=True)
    dim = 256
    
    # Coordinate grid
    y, x = np.ogrid[:dim, :dim]
    
    # Normal brain tissue ellipse
    brain_center = (128, 128)
    brain_radius = 100
    brain_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 <= brain_radius**2
    
    # Base normal brain image (no hyperintense tumor region)
    data = np.zeros((dim, dim), dtype=np.float32)
    
    np.random.seed(101)
    normal_tissue = np.random.normal(loc=110, scale=12, size=(dim, dim)).astype(np.float32)
    data[brain_mask] = np.clip(normal_tissue[brain_mask], 30, 160)
    
    # Save as PNG image
    img_uint8 = data.astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    output_path = os.path.join('data', 'sample_normal_mri.png')
    img.save(output_path)
    print(f"Normal healthy brain MRI slice saved to: {output_path} (Dimensions: {img.size})")

if __name__ == "__main__":
    generate_sample_normal_2d()
