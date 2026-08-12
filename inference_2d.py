import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

from train_2d_unet import LightweightUNet2D

_trained_model_instance = None
CHECKPOINT_PATH = os.path.join("models", "brain_tumor_unet_2d.pth")

def load_trained_model() -> LightweightUNet2D:
    """
    Loads the trained PyTorch 2D UNet model checkpoint from disk ('models/brain_tumor_unet_2d.pth').
    Sets the model strictly to evaluation mode (eval()) for real inference.
    """
    global _trained_model_instance
    if _trained_model_instance is None:
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"Trained model checkpoint missing at: {CHECKPOINT_PATH}")
            
        model = LightweightUNet2D(in_channels=1, out_channels=1)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        _trained_model_instance = model
        print(f"Loaded trained 2D UNet checkpoint from '{CHECKPOINT_PATH}'.")
    return _trained_model_instance

def validate_brain_mri(raw_image: np.ndarray) -> Tuple[bool, str]:
    """
    Basic input-quality validation check to verify whether an input image resembles a 2D brain MRI scan.
    Checks outer border darkness and tissue intensity variance.
    """
    h, w = raw_image.shape
    border_width = 12
    
    top_border = raw_image[:border_width, :]
    bottom_border = raw_image[-border_width:, :]
    left_border = raw_image[:, :border_width]
    right_border = raw_image[:, -border_width:]
    
    border_pixels = np.concatenate([top_border.ravel(), bottom_border.ravel(), left_border.ravel(), right_border.ravel()])
    mean_border_intensity = np.mean(border_pixels)
    
    if mean_border_intensity > 55.0:
        return False, f"Invalid input: Image border is too bright ({mean_border_intensity:.1f} avg intensity). Brain MRIs require a dark background."
        
    brain_pixels = np.sum(raw_image > 20)
    foreground_ratio = brain_pixels / (h * w)
    
    if foreground_ratio < 0.10 or foreground_ratio > 0.85:
        return False, f"Invalid input: Foreground tissue ratio ({foreground_ratio*100:.1f}%) is outside expected MRI structural limits (10% - 85%)."
        
    fg_std = np.std(raw_image[raw_image > 20]) if brain_pixels > 0 else 0
    if fg_std < 5.0:
        return False, "Invalid input: Image lacks expected MRI tissue intensity variance."
        
    return True, "Valid Brain MRI Scan"

def preprocess_2d(image_input) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Standardized preprocessing pipeline identical to training:
    1. Grayscale conversion ('L')
    2. Resize to (256, 256)
    3. Min-Max Intensity Normalization (0.0 to 1.0)
    4. 4D Float Tensor conversion (1, 1, 256, 256)
    """
    img = Image.open(image_input).convert('L')
    img_resized = img.resize((256, 256), Image.Resampling.BILINEAR)
    
    raw_data = np.array(img_resized, dtype=np.float32)
    normalized_data = raw_data / 255.0 # Min-Max Scaling (0.0 - 1.0) matching dataset training
    
    tensor = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).float()
    return tensor, raw_data

def predict_2d(tensor: torch.Tensor, raw_image: np.ndarray, prob_threshold: float = 0.50, min_pixel_threshold: int = 20) -> Dict[str, Any]:
    """
    Executes genuine model forward pass using the loaded trained checkpoint ('models/brain_tumor_unet_2d.pth').
    Derives all tumor metrics directly from the model's actual output probability mask.
    """
    start_time = time.perf_counter()
    
    # 1. Basic Quality & Structural Validation
    is_valid, validation_msg = validate_brain_mri(raw_image)
    if not is_valid:
        end_time = time.perf_counter()
        return {
            "is_valid_mri": False,
            "validation_error": validation_msg,
            "checkpoint_path": CHECKPOINT_PATH,
            "model_architecture": "LightweightUNet2D (PyTorch)",
            "model_called": False,
            "model_status": "Skipped (Invalid Non-MRI Input)",
            "tensor_shape": list(tensor.shape),
            "mask": np.zeros_like(raw_image, dtype=np.uint8),
            "tumor_detected": False,
            "tumor_pixel_count": 0,
            "confidence": None,
            "execution_time_ms": round((end_time - start_time) * 1000.0, 2),
            "prob_threshold": prob_threshold,
            "min_pixel_threshold": min_pixel_threshold
        }

    # 2. Genuine PyTorch Model Inference
    model = load_trained_model()
    
    with torch.no_grad():
        logits = model(tensor) # Forward pass through trained weights
        probabilities = torch.sigmoid(logits) # Sigmoid activation probability map
        
        prob_map = probabilities[0, 0, :, :].cpu().numpy()
        
        # Apply documented segmentation probability threshold (PROB_THRESHOLD = 0.50)
        binary_mask = (prob_map >= prob_threshold).astype(np.uint8)
        tumor_pixel_count = int(np.sum(binary_mask == 1))
        
        # Decision thresholding
        if tumor_pixel_count < min_pixel_threshold:
            binary_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            tumor_pixel_count = 0
            confidence = None
            tumor_detected = False
        else:
            tumor_detected = True
            tumor_probs_in_mask = prob_map[binary_mask == 1]
            confidence = float(np.mean(tumor_probs_in_mask)) if len(tumor_probs_in_mask) > 0 else float(prob_threshold)
            
    end_time = time.perf_counter()
    execution_time_ms = round((end_time - start_time) * 1000.0, 2)
    
    return {
        "is_valid_mri": True,
        "validation_error": None,
        "checkpoint_path": CHECKPOINT_PATH,
        "model_architecture": "LightweightUNet2D (PyTorch)",
        "model_called": True,
        "model_status": f"Loaded Checkpoint '{CHECKPOINT_PATH}'",
        "tensor_shape": list(tensor.shape),
        "mask": binary_mask,
        "tumor_detected": tumor_detected,
        "tumor_pixel_count": tumor_pixel_count,
        "confidence": confidence,
        "execution_time_ms": execution_time_ms,
        "prob_threshold": prob_threshold,
        "min_pixel_threshold": min_pixel_threshold
    }

if __name__ == "__main__":
    print("--- 1. Testing Trained Model on Known Tumor MRI ---")
    t1, r1 = preprocess_2d("data/sample_glioma.png")
    p1 = predict_2d(t1, r1)
    print(f"Valid: {p1['is_valid_mri']} | Model Called: {p1['model_called']} | Detected: {p1['tumor_detected']} | Pixels: {p1['tumor_pixel_count']} | Time: {p1['execution_time_ms']} ms | Conf: {p1['confidence']}")
    
    print("\n--- 2. Testing Trained Model on Known Non-Tumor MRI ---")
    t2, r2 = preprocess_2d("data/sample_normal_mri.png")
    p2 = predict_2d(t2, r2)
    print(f"Valid: {p2['is_valid_mri']} | Model Called: {p2['model_called']} | Detected: {p2['tumor_detected']} | Pixels: {p2['tumor_pixel_count']} | Time: {p2['execution_time_ms']} ms | Conf: {p2['confidence']}")
    
    print("\n--- 3. Testing Trained Model on Non-MRI Image ---")
    t3, r3 = preprocess_2d("data/sample_non_mri_bottle.png")
    p3 = predict_2d(t3, r3)
    print(f"Valid: {p3['is_valid_mri']} | Model Called: {p3['model_called']} | Error: {p3['validation_error']}")
