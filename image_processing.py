import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def apply_high_contrast(image_array: np.ndarray) -> np.ndarray:
    """
    Applies adaptive contrast stretching using 2nd to 98th percentile intensity scaling.
    Enhances contrast for medical image analysis on low-contrast MRI scans.
    """
    vmin, vmax = np.percentile(image_array, (2, 98))
    if vmax > vmin:
        scaled = np.clip((image_array - vmin) / (vmax - vmin), 0.0, 1.0) * 255.0
    else:
        scaled = image_array
    return scaled.astype(np.float32)

def render_segmentation_figure(
    raw_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    show_overlay: bool = True,
    high_contrast: bool = False,
    show_contours: bool = True,
    title: str = "2D MRI Tumor Segmentation Overlay"
) -> plt.Figure:
    """
    Renders a high-resolution Matplotlib figure with high contrast adjustments,
    semi-transparent red tumor mask overlay, and yellow perimeter contour lines.
    """
    display_image = apply_high_contrast(raw_image) if high_contrast else raw_image
    
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    ax.imshow(display_image, cmap='gray', origin='upper')
    
    if show_overlay and np.any(mask == 1):
        # 1. Semi-transparent red mask overlay
        tumor_overlay = np.ma.masked_where(mask != 1, mask)
        ax.imshow(tumor_overlay, cmap='Reds', alpha=alpha, vmin=0, vmax=1, origin='upper')
        
        # 2. Bright yellow perimeter contour lines
        if show_contours:
            ax.contour(mask, levels=[0.5], colors='#FACC15', linewidths=1.5, origin='upper')
            
        ax.set_title(title, color='#F87171', fontsize=12, fontweight='bold')
    elif show_overlay:
        ax.set_title("No Tumor Regions Detected", color='#94A3B8', fontsize=12)
    else:
        ax.set_title("Overlay Hidden", color='#94A3B8', fontsize=12)
        
    ax.axis('off')
    fig.tight_layout()
    return fig

def export_overlay_as_png(raw_image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> bytes:
    """
    Exports the segmented MRI image with red tumor overlay as a PNG byte buffer for user download.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.imshow(raw_image, cmap='gray', origin='upper')
    if np.any(mask == 1):
        tumor_overlay = np.ma.masked_where(mask != 1, mask)
        ax.imshow(tumor_overlay, cmap='Reds', alpha=alpha, vmin=0, vmax=1, origin='upper')
        ax.contour(mask, levels=[0.5], colors='yellow', linewidths=1.5, origin='upper')
        
    ax.axis('off')
    ax.set_title("Brain Tumor Segmentation Result", color='black', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
