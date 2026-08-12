import numpy as np
from inference import load_and_preprocess, predict_segmentation

def calculate_tumor_volume(mask_array: np.ndarray, affine: np.ndarray) -> float:
    """
    Calculates total tumor volume in cubic centimeters (cm³) from the segmentation mask and NIfTI affine matrix.
    """
    # Voxel dimensions in mm from 4x4 affine matrix
    voxel_dims = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    dx, dy, dz = voxel_dims[0], voxel_dims[1], voxel_dims[2]
    
    voxel_vol_mm3 = dx * dy * dz
    tumor_voxel_count = np.sum(mask_array == 1)
    
    # 1 cm³ = 1000 mm³
    tumor_vol_cm3 = float((tumor_voxel_count * voxel_vol_mm3) / 1000.0)
    return tumor_vol_cm3

def get_peak_tumor_slice(mask_array: np.ndarray, axis: int = 2):
    """
    Calculates tumor voxel area per slice across the specified axis (default: axial z-axis).
    Returns the slice index with maximum tumor area and a list of all slice indices containing tumor voxels.
    """
    num_slices = mask_array.shape[axis]
    slice_areas = []
    
    for i in range(num_slices):
        if axis == 0:
            slice_mask = mask_array[i, :, :]
        elif axis == 1:
            slice_mask = mask_array[:, i, :]
        else: # axis == 2
            slice_mask = mask_array[:, :, i]
            
        area = np.sum(slice_mask == 1)
        slice_areas.append(area)
        
    slice_areas = np.array(slice_areas)
    
    if np.max(slice_areas) == 0:
        peak_slice_idx = 0
        tumor_slices = []
    else:
        peak_slice_idx = int(np.argmax(slice_areas))
        tumor_slices = [int(idx) for idx in np.where(slice_areas > 0)[0]]
        
    return peak_slice_idx, tumor_slices

if __name__ == "__main__":
    nifti_path = "data/sample_brain.nii.gz"
    print(f"Loading {nifti_path}...")
    tensor, affine = load_and_preprocess(nifti_path)
    
    print("Predicting segmentation mask...")
    mask = predict_segmentation(tensor)
    
    volume_cm3 = calculate_tumor_volume(mask, affine)
    peak_slice, active_slices = get_peak_tumor_slice(mask, axis=2)
    
    print(f"\n--- Clinical Metrics Results ---")
    print(f"Total Tumor Volume: {volume_cm3:.4f} cm³")
    print(f"Peak Tumor Axial Slice Index: {peak_slice}")
    print(f"Slices containing tumor voxels: {active_slices if active_slices else 'None'}")
