import numpy as np
from typing import Dict, Any

def analyze_tumor_metrics(
    mask_array: np.ndarray,
    raw_image_array: np.ndarray,
    confidence: float = None,
    tumor_type: str = "No Tumor Detected",
    type_confidence: float = 1.0
) -> Dict[str, Any]:
    """
    Calculates quantitative clinical metrics and particular tumor subtype classification details.
    """
    tumor_pixels = int(np.sum(mask_array == 1))
    brain_pixels = int(np.sum(raw_image_array > 20))
    
    if brain_pixels > 0 and tumor_pixels > 0:
        coverage_pct = float((tumor_pixels / brain_pixels) * 100.0)
    else:
        coverage_pct = 0.0
        
    tumor_detected = tumor_pixels > 0
    
    if confidence is not None and tumor_detected:
        confidence_str = f"{confidence * 100:.1f}%"
    else:
        confidence_str = "Not available"
        
    type_conf_str = f"{type_confidence * 100:.1f}%" if tumor_detected else "100%"
    
    return {
        "tumor_detected": tumor_detected,
        "tumor_detected_label": "YES" if tumor_detected else "NO",
        "tumor_type": tumor_type if tumor_detected else "Healthy Brain",
        "tumor_type_confidence_str": type_conf_str,
        "tumor_area_pixels": tumor_pixels,
        "brain_area_pixels": brain_pixels,
        "brain_coverage_pct": round(coverage_pct, 2),
        "confidence_score": confidence_str,
        "raw_confidence": confidence
    }

if __name__ == "__main__":
    dummy_mask = np.zeros((256, 256), dtype=np.uint8)
    dummy_mask[100:130, 100:130] = 1
    dummy_raw = np.full((256, 256), 100, dtype=np.float32)
    
    metrics = analyze_tumor_metrics(dummy_mask, dummy_raw, confidence=0.924, tumor_type="Pituitary Tumor", type_confidence=0.942)
    print("--- Particular Subtype Output ---")
    print(f"Tumor Type: {metrics['tumor_type']}")
    print(f"Subtype Confidence: {metrics['tumor_type_confidence_str']}")
