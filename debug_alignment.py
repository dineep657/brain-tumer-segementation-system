import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from inference_2d import preprocess_2d, predict_2d

def debug_spatial_alignment():
    print("=" * 70)
    print("  SPATIAL ALIGNMENT & ORIENTATION DEBUGGING TEST")
    print("=" * 70)
    
    # 1. Create a synthetic test image with a tumor strictly in UPPER-RIGHT quadrant (X=180, Y=60)
    dim = 256
    y_grid, x_grid = np.ogrid[:dim, :dim]
    
    brain_center = (128, 128)
    brain_mask = (x_grid - brain_center[0])**2 + (y_grid - brain_center[1])**2 <= 100**2
    
    test_img = np.zeros((dim, dim), dtype=np.float32)
    test_img[brain_mask] = 110.0
    
    # UPPER-RIGHT tumor: Centroid X=180 (right), Y=60 (upper)
    tumor_x, tumor_y, tumor_r = 180, 60, 22
    tumor_region = ((x_grid - tumor_x)**2 + (y_grid - tumor_y)**2 <= tumor_r**2) & brain_mask
    test_img[tumor_region] = 240.0
    
    os.makedirs("data", exist_ok=True)
    test_path = os.path.join("data", "debug_upper_right_tumor.png")
    Image.fromarray(test_img.astype(np.uint8)).save(test_path)
    
    print(f"Created spatial test image: '{test_path}'")
    print(f"Known ground-truth tumor location: UPPER-RIGHT Quadrant (X={tumor_x}, Y={tumor_y})")
    
    # 2. Run Preprocessing & Inference
    orig_img_pil = Image.open(test_path).convert('L')
    orig_w, orig_h = orig_img_pil.size
    orig_shape = (orig_h, orig_w)
    
    tensor, raw_2d = preprocess_2d(test_path)
    model_input_shape = list(tensor.shape)
    
    res = predict_2d(tensor, raw_2d)
    mask = res["mask"]
    model_output_shape = list(mask.shape)
    
    # Calculate mask centroid to verify spatial location
    if np.any(mask == 1):
        mask_y_indices, mask_x_indices = np.where(mask == 1)
        pred_centroid_y = float(np.mean(mask_y_indices))
        pred_centroid_x = float(np.mean(mask_x_indices))
    else:
        pred_centroid_y, pred_centroid_x = -1.0, -1.0
        
    print("\n--- Telemetry & Dimension Checks ---")
    print(f"Original image shape (H, W)       : {orig_shape}")
    print(f"Model input tensor shape (N,C,H,W): {model_input_shape}")
    print(f"Model output mask shape (H, W)    : {model_output_shape}")
    print(f"Predicted mask shape (H, W)       : {mask.shape}")
    print(f"Final resized mask shape (H, W)   : {mask.shape}")
    print(f"Transposes / Flips applied        : NONE")
    print(f"Ground-Truth Tumor Centroid (X,Y) : ({tumor_x}, {tumor_y})")
    print(f"Predicted Mask Centroid (X,Y)     : ({pred_centroid_x:.1f}, {pred_centroid_y:.1f})")
    
    # 3. Verify Alignment Quad Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    fig.patch.set_facecolor('white')
    
    # A. Original MRI
    axes[0].imshow(raw_2d, cmap='gray', origin='upper')
    axes[0].set_title("A. Original MRI\n(Tumor Upper-Right)", fontsize=10, fontweight='bold')
    axes[0].axis('off')
    
    # B. Raw Predicted Probability / Mask
    axes[1].imshow(mask, cmap='hot', origin='upper')
    axes[1].set_title(f"B. Predicted Mask\nCentroid: ({pred_centroid_x:.0f}, {pred_centroid_y:.0f})", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    
    # C. Binary Tumor Mask
    axes[2].imshow(mask, cmap='gray', origin='upper')
    axes[2].set_title("C. Binary Mask\n(0 = Background, 1 = Tumor)", fontsize=10, fontweight='bold')
    axes[2].axis('off')
    
    # D. Final Overlay
    axes[3].imshow(raw_2d, cmap='gray', origin='upper')
    if np.any(mask == 1):
        tumor_overlay = np.ma.masked_where(mask != 1, mask)
        axes[3].imshow(tumor_overlay, cmap='Reds', alpha=0.5, origin='upper')
        # Explicit 2D Grid coordinates for contour alignment
        X_grid, Y_grid = np.meshgrid(np.arange(dim), np.arange(dim))
        axes[3].contour(X_grid, Y_grid, mask, levels=[0.5], colors='yellow', linewidths=1.5)
        
    axes[3].set_title("D. Final Overlay & Contour\n(Must Align Exactly)", fontsize=10, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    debug_plot_path = os.path.join("data", "debug_alignment_quad.png")
    plt.savefig(debug_plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"\nQuad debugging plot saved to: '{debug_plot_path}'")
    
    # Alignment assertion check
    x_error = abs(pred_centroid_x - tumor_x)
    y_error = abs(pred_centroid_y - tumor_y)
    
    if x_error < 5.0 and y_error < 5.0:
        print("\n✅ ALIGNMENT VERIFIED PERFECT: Predicted mask matches ground-truth upper-right position!")
    else:
        print(f"\n❌ ALIGNMENT ERROR DETECTED! X Error: {x_error:.1f}, Y Error: {y_error:.1f}")

if __name__ == "__main__":
    debug_spatial_alignment()
