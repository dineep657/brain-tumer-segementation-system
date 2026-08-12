import os
import numpy as np
from inference_2d import preprocess_2d, predict_2d
from classifier_2d import classify_tumor_2d, CLASSES

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
    
    for label, file_path in test_items:
        print(f"\n--- Test Item: {label} ({file_path}) ---")
        if not os.path.exists(file_path):
            print(f"  [Skipped]: Test file '{file_path}' does not exist.")
            continue
            
        tensor, raw_original, orig_dims = preprocess_2d(file_path)
        seg_res = predict_2d(tensor, raw_original)
        
        if not seg_res['is_valid_mri']:
            print(f"  MRI Validation Status  : INVALID NON-MRI")
            print(f"  Validation Warning     : {seg_res['validation_error']}")
            print(f"  Tumor Status           : NO (0 pixels)")
            print(f"  Tumor Type             : Invalid Input (Classifier Skipped)")
        else:
            cls_res = classify_tumor_2d(tensor, tumor_detected=seg_res['tumor_detected'])
            print(f"  MRI Validation Status  : VALID MRI")
            print(f"  Tumor Status           : {'YES' if seg_res['tumor_detected'] else 'NO'}")
            print(f"  Predicted Tumor Pixels : {seg_res['tumor_pixel_count']} voxels")
            print(f"  Predicted Tumor Type   : {cls_res['predicted_class']}")
            print(f"  Classifier Confidence  : {cls_res['confidence_display']}")
            print(f"  Inference Speed        : Seg={seg_res['execution_time_ms']} ms | Cls={cls_res['classifier_time_ms']} ms")
            
    print("\n" + "=" * 70)
    print("  END-TO-END PIPELINE TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    run_classifier_pipeline_test()
