import os
import torch
import numpy as np

from inference_2d import preprocess_2d, predict_2d
from classifier_2d import classify_tumor_2d

def run_classifier_pipeline_test():
    print("=" * 70)
    print("  TESTING END-TO-END UNET SEGMENTATION + CNN CLASSIFIER PIPELINE")
    print("=" * 70)
    
    test_items = [
        ("Glioma Tumor Sample", "data/sample_glioma.png"),
        ("Meningioma Tumor Sample", "data/sample_meningioma.png"),
        ("Pituitary Tumor Sample", "data/sample_pituitary.png"),
        ("Healthy Normal MRI", "data/sample_normal_mri.png"),
        ("Non-Brain Image Test (Plastic Bottle)", "data/sample_non_mri_bottle.png")
    ]
    
    for label, filepath in test_items:
        print(f"\n--- Test Item: {label} ({filepath}) ---")
        if not os.path.exists(filepath):
            print(f"  File missing: {filepath}")
            continue
            
        # 1. Preprocessing & UNet Segmentation
        tensor, raw_img, orig_dim = preprocess_2d(filepath)
        seg_res = predict_2d(tensor, raw_img)
        
        if not seg_res["is_valid_mri"]:
            print(f"  MRI Validation Status  : INVALID NON-MRI")
            print(f"  Validation Warning     : {seg_res['validation_error']}")
            print(f"  Tumor Status           : NO ({seg_res['tumor_pixel_count']} pixels)")
            print(f"  Tumor Type             : Invalid Input (Classifier Skipped)")
            continue
            
        print(f"  MRI Validation Status  : VALID MRI")
        print(f"  Tumor Status           : {'YES' if seg_res['tumor_detected'] else 'NO'}")
        print(f"  Predicted Tumor Pixels : {seg_res['tumor_pixel_count']} voxels")
        
        # 2. Multi-Class Classification
        cls_res = classify_tumor_2d(tensor, seg_res["tumor_detected"], min_confidence_threshold=0.35)
        
        conf_str = f"{cls_res['confidence']*100:.1f}%" if cls_res['confidence'] is not None else "N/A"
        print(f"  Predicted Tumor Type   : {cls_res['tumor_type']}")
        print(f"  Classifier Confidence  : {conf_str}")
        print(f"  Inference Speed        : Seg={seg_res['execution_time_ms']} ms | Cls={cls_res['execution_time_ms']} ms")
        
    print("\n" + "=" * 70)
    print("  END-TO-END PIPELINE TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_classifier_pipeline_test()
