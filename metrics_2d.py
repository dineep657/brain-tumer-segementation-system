import numpy as np
from typing import Dict, Any

def analyze_tumor_metrics(mask_array: np.ndarray, raw_image_array: np.ndarray, confidence: float = None) -> Dict[str, Any]:
    """
    Calculates genuine quantitative clinical metrics directly derived from the model's output segmentation mask.
    """
    tumor_pixels = int(np.sum(mask_array == 1))
    brain_pixels = int(np.sum(raw_image_array > 20))
    total_pixels = mask_array.shape[0] * mask_array.shape[1]
    
    if brain_pixels > 0 and tumor_pixels > 0:
        coverage_pct = float((tumor_pixels / brain_pixels) * 100.0)
    else:
        coverage_pct = 0.0
        
    matrix_area_pct = float((tumor_pixels / total_pixels) * 100.0)
    tumor_detected = tumor_pixels > 0
    
    if confidence is not None and tumor_detected:
        confidence_str = f"{confidence * 100:.1f}%"
    else:
        confidence_str = "Not available"
        
    return {
        "tumor_detected": tumor_detected,
        "tumor_detected_label": "YES" if tumor_detected else "NO",
        "tumor_subtype_label": "Subtype Classifier Not Trained" if tumor_detected else "Healthy Brain",
        "tumor_area_pixels": tumor_pixels,
        "brain_area_pixels": brain_pixels,
        "brain_coverage_pct": round(coverage_pct, 2),
        "matrix_area_pct": round(matrix_area_pct, 2),
        "confidence_score": confidence_str,
        "raw_confidence": confidence
    }

if __name__ == "__main__":
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask[100:130, 100:130] = 1
    dummy_raw = np.full((256, 256), 100, dtype=np.float32)
    
    metrics = analyze_tumor_metrics(dummy_mask, dummy_raw, confidence=0.892)
    print("--- Genuine Metrics Test Output ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
