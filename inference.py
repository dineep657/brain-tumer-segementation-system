import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import SegResNet

def load_and_preprocess(nifti_path: str):
    """
    Loads a .nii.gz file using nibabel, applies Z-score intensity normalization,
    and returns a 5D PyTorch float tensor (1, 1, D, H, W) along with the NIfTI affine matrix.
    """
    nifti_img = nib.load(nifti_path)
    affine = nifti_img.affine
    data = nifti_img.get_fdata().astype(np.float32)
    
    # Z-score intensity normalization
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / (std + 1e-8)
    
    # Convert to 5D PyTorch tensor: (Batch=1, Channel=1, Depth, Height, Width)
    tensor = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).float()
    return tensor, affine

def predict_segmentation(tensor: torch.Tensor) -> np.ndarray:
    """
    Runs forward inference on the input 5D tensor using MONAI SegResNet.
    Returns a 3D binary/class segmentation mask numpy array of shape (D, H, W).
    """
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2
    )
    model.eval()
    
    with torch.no_grad():
        logits = model(tensor)
        preds = torch.argmax(logits, dim=1) # Shape: (1, D, H, W)
        mask = preds.squeeze(0).cpu().numpy() # Shape: (D, H, W)
        
    return mask

if __name__ == "__main__":
    nifti_file = "data/sample_brain.nii.gz"
    print(f"Loading and preprocessing {nifti_file}...")
    tensor, affine = load_and_preprocess(nifti_file)
    print(f"Input tensor shape: {tensor.shape}")
    
    print("Running SegResNet model inference...")
    mask = predict_segmentation(tensor)
    
    print(f"Output mask shape: {mask.shape}")
    print(f"Unique predicted class values: {np.unique(mask)}")
