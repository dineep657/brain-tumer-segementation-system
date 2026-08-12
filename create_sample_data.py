import os
import numpy as np
import nibabel as nib

def generate_sample_brain():
    os.makedirs('data', exist_ok=True)
    
    # 64x64x64 volume
    dim = 64
    data = np.zeros((dim, dim, dim), dtype=np.float32)
    
    # Coordinate grid
    z, y, x = np.ogrid[:dim, :dim, :dim]
    
    # 1. Simulating brain tissue (center 32,32,32, radius 25)
    brain_center = (32, 32, 32)
    brain_radius = 25
    brain_mask = (x - brain_center[0])**2 + (y - brain_center[1])**2 + (z - brain_center[2])**2 <= brain_radius**2
    
    # Add brain intensity with noise
    np.random.seed(42)
    brain_noise = np.random.normal(loc=0.5, scale=0.08, size=(dim, dim, dim)).astype(np.float32)
    data[brain_mask] = np.clip(brain_noise[brain_mask], 0.1, 0.8)
    
    # 2. Simulating dummy spherical tumor region (center 36, 28, 36, radius 8)
    tumor_center = (36, 28, 36)
    tumor_radius = 8
    tumor_mask = (x - tumor_center[0])**2 + (y - tumor_center[1])**2 + (z - tumor_center[2])**2 <= tumor_radius**2
    
    # Add elevated tumor intensity with noise
    tumor_noise = np.random.normal(loc=0.95, scale=0.05, size=(dim, dim, dim)).astype(np.float32)
    data[tumor_mask] = np.clip(tumor_noise[tumor_mask], 0.8, 1.0)
    
    # Save as NIfTI (.nii.gz) file
    output_path = os.path.join('data', 'sample_brain.nii.gz')
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)
    
    print(f"Sample brain volume saved to {output_path}")
    print(f"Volume shape: {data.shape}")

if __name__ == "__main__":
    generate_sample_brain()
