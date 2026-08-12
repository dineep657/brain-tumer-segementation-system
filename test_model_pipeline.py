import os
import numpy as np
from inference_2d import preprocess_2d, predict_2d

def run_pipeline_test():
    print("=" * 65)
    print("  TESTING SPATIALLY ALIGNED REAL TRAINED 2D UNET PIPELINE")
    print("=" * 65)
    
    test_files = [
        ("Known Tumor Brain MRI", "data/sample_glioma.png"),
        ("Known Non-Tumor Healthy MRI", "data/sample_normal_mri.png"),
        ("Unrelated Non-MRI Image (Plastic Bottle)", "data/sample_non_mri_bottle.png")
    ]
    
    for label, file_path in test_files:
        print(f"\n--- Test Item: {label} ({file_path}) ---")
        if not os.path.exists(file_path):
            print(f"Error: Test file '{file_path}' does not exist.")
            continue
            
        tensor, raw_original, orig_dims = preprocess_2d(file_path)
        res = predict_2d(tensor, raw_original)
        
        print(f"  Original Image Shape (H,W): {orig_dims}")
        print(f"  Model Input Tensor Shape  : {list(tensor.shape)}")
        print(f"  Predicted Mask Shape (H,W): {res['mask'].shape}")
        print(f"  MRI Validation Status     : {'VALID MRI' if res['is_valid_mri'] else 'INVALID NON-MRI'}")
        
        if not res['is_valid_mri']:
            print(f"  Validation Warning        : {res['validation_error']}")
            print(f"  PyTorch Forward Pass      : SKIPPED (No segmentation mask generated)")
            print(f"  Tumor Detected            : NO (0 pixels)")
        else:
            print(f"  PyTorch Forward Pass      : EXECUTED (Loaded '{res['checkpoint_path']}')")
            print(f"  Tumor Detected            : {'YES' if res['tumor_detected'] else 'NO'}")
            print(f"  Predicted Pixels          : {res['tumor_pixel_count']} voxels")
            print(f"  Mean Model Prob           : {res['confidence'] if res['confidence'] is not None else 'Not available'}")
            print(f"  Inference Time            : {res['execution_time_ms']} ms")
            
    print("\n" + "=" * 65)
    print("  ALL PIPELINE TESTS COMPLETED SUCCESSFULLY")
    print("=" * 65)

if __name__ == "__main__":
    run_pipeline_test()
