import os
import numpy as np
from PIL import Image

def generate_non_mri_sample():
    os.makedirs('data', exist_ok=True)
    dim = 256
    
    # Create a non-brain image (e.g., bright background with a bottle shape or rectangular object)
    data = np.full((dim, dim), 200, dtype=np.uint8) # Bright background (unlike MRI black border)
    
    # Rectangular object in center (bottle-like shape)
    data[50:200, 90:166] = 50
    data[30:50, 110:146] = 80
    
    img = Image.fromarray(data)
    output_path = os.path.join('data', 'sample_non_mri_bottle.png')
    img.save(output_path)
    print(f"Non-MRI test sample (plastic bottle photo simulation) saved to: {output_path}")

if __name__ == "__main__":
    generate_non_mri_sample()
