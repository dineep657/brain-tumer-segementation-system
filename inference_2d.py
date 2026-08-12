import os
import time
import torch
import torch.nn as nn
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
    border_width = min(12, h // 10, w // 10)
    
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

def preprocess_2d(image_input) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """
    Standardized preprocessing pipeline matching dataset training:
    1. Open PIL image as grayscale ('L')
    2. Store original dimensions (orig_h, orig_w)
    3. Resize image to (256, 256)
    4. Min-Max Intensity Scaling (0.0 to 1.0)
    5. Convert to 4D Float Tensor (1, 1, 256, 256)
    """
    img_pil = Image.open(image_input).convert('L')
    orig_w, orig_h = img_pil.size
    
    raw_original = np.array(img_pil, dtype=np.float32)
    
    img_256 = img_pil.resize((256, 256), Image.Resampling.BILINEAR)
    img_arr = np.array(img_256, dtype=np.float32)
    
    min_val, max_val = img_arr.min(), img_arr.max()
    if max_val > min_val:
        normalized_data = (img_arr - min_val) / (max_val - min_val + 1e-8)
    else:
        normalized_data = img_arr / 255.0
        
    tensor = torch.from_numpy(normalized_data).unsqueeze(0).unsqueeze(0).float()
    return tensor, raw_original, (orig_h, orig_w)

def predict_2d(tensor: torch.Tensor, raw_original_image: np.ndarray, prob_threshold: float = 0.50, min_pixel_threshold: int = 20) -> Dict[str, Any]:
    """
    Executes PyTorch forward pass using the loaded trained checkpoint ('models/brain_tumor_unet_2d.pth').
    Derives all tumor metrics directly from the model's actual output probability mask.
    Resizes binary mask back to original image dimensions (orig_h, orig_w) using Nearest-Neighbor interpolation.
    """
    start_time = time.perf_counter()
    orig_h, orig_w = raw_original_image.shape
    
    # 1. Basic Quality & Structural Validation
    is_valid, validation_msg = validate_brain_mri(raw_original_image)
    if not is_valid:
        end_time = time.perf_counter()
        return {
            "is_valid_mri": False,
            "validation_error": validation_msg,
            "checkpoint_path": CHECKPOINT_PATH,
            "model_architecture": "LightweightUNet2D (PyTorch)",
            "model_called": False,
            "tensor_shape": list(tensor.shape),
            "mask": np.zeros((orig_h, orig_w), dtype=np.uint8),
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
        logits = model(tensor) # Forward pass
        probabilities = torch.sigmoid(logits) # Sigmoid probability map (1, 1, 256, 256)
        
        prob_map_256 = probabilities[0, 0, :, :].cpu().numpy()
        binary_mask_256 = (prob_map_256 >= prob_threshold).astype(np.uint8)
        
        # Resize binary mask back to original image dimensions (orig_h, orig_w) using Nearest-Neighbor
        if (orig_h, orig_w) != (256, 256):
            mask_pil = Image.fromarray(binary_mask_256).resize((orig_w, orig_h), Image.Resampling.NEAREST)
            final_mask = np.array(mask_pil, dtype=np.uint8)
            prob_pil = Image.fromarray((prob_map_256 * 255.0).astype(np.uint8)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            final_prob_map = np.array(prob_pil, dtype=np.float32) / 255.0
        else:
            final_mask = binary_mask_256
            final_prob_map = prob_map_256
            
        tumor_pixel_count = int(np.sum(final_mask == 1))
        
        if tumor_pixel_count < min_pixel_threshold:
            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            tumor_pixel_count = 0
            confidence = None
            tumor_detected = False
        else:
            tumor_detected = True
            tumor_probs_in_mask = final_prob_map[final_mask == 1]
            confidence = float(np.mean(tumor_probs_in_mask)) if len(tumor_probs_in_mask) > 0 else float(prob_threshold)
            
    end_time = time.perf_counter()
    execution_time_ms = round((end_time - start_time) * 1000.0, 2)
    
    return {
        "is_valid_mri": True,
        "validation_error": None,
        "checkpoint_path": CHECKPOINT_PATH,
        "model_architecture": "LightweightUNet2D (PyTorch)",
        "model_called": True,
        "tensor_shape": list(tensor.shape),
        "mask": final_mask,
        "tumor_detected": tumor_detected,
        "tumor_pixel_count": tumor_pixel_count,
        "confidence": confidence,
        "execution_time_ms": execution_time_ms,
        "prob_threshold": prob_threshold,
        "min_pixel_threshold": min_pixel_threshold
    }
