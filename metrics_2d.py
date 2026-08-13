import numpy as np
from typing import Dict, Any

def analyze_tumor_metrics(
    mask_array: np.ndarray,
    raw_image_array: np.ndarray,
    seg_confidence: float = None,
    cls_result: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Calculates genuine quantitative clinical metrics derived from actual model outputs.
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
    
    if seg_confidence is not None and tumor_detected:
        seg_confidence_str = f"{seg_confidence * 100:.1f}%"
    else:
        seg_confidence_str = "Not available"
        
    if cls_result is not None:
        tumor_type = cls_result.get("tumor_type", cls_result.get("predicted_class", "No Tumor"))
        cls_confidence = cls_result.get("confidence")
        if cls_confidence is not None and tumor_type != "No Tumor" and tumor_type != "Invalid Input":
            cls_confidence_str = f"{cls_confidence * 100:.1f}%"
        else:
            cls_confidence_str = cls_result.get("confidence_display", "N/A")
    else:
        tumor_type = "No Tumor" if not tumor_detected else "Unknown"
        cls_confidence_str = "N/A"
        
    return {
        "tumor_detected": tumor_detected,
        "tumor_detected_label": "YES" if tumor_detected else "NO",
        "tumor_subtype_label": tumor_type,
        "classification_confidence": cls_confidence_str,
        "tumor_area_pixels": tumor_pixels,
        "brain_area_pixels": brain_pixels,
        "brain_coverage_pct": round(coverage_pct, 2),
        "matrix_area_pct": round(matrix_area_pct, 2),
        "segmentation_confidence_score": seg_confidence_str
    }
