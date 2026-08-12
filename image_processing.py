import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def render_segmentation_figure(
    raw_image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    show_overlay: bool = True,
    high_contrast: bool = False,
    show_contours: bool = True,
    title: str = "MRI Segmentation Result"
) -> plt.Figure:
    """
    Renders a Matplotlib figure ensuring 100% spatial alignment between the original MRI image,
    the semi-transparent red tumor mask overlay, and the yellow perimeter contour lines.
    
    CRITICAL FIX: Uses explicit 2D meshgrid coordinates (X_grid, Y_grid) for ax.contour()
    to prevent Matplotlib from transposing row/column indices relative to ax.imshow().
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    # 1. High Contrast Scaling
    display_image = raw_image.copy()
    if high_contrast:
        p2, p98 = np.percentile(display_image, (2, 98))
        if p98 > p2:
            display_image = np.clip((display_image - p2) / (p98 - p2), 0, 1) * 255.0

    # 2. Render Base Grayscale Image
    ax.imshow(display_image, cmap='gray', origin='upper')
    
    h, w = mask.shape
    
    # 3. Render Semi-Transparent Red Mask Overlay
    if show_overlay and np.any(mask == 1):
        tumor_overlay = np.ma.masked_where(mask != 1, mask)
        ax.imshow(tumor_overlay, cmap='Reds', alpha=alpha, vmin=0, vmax=1, origin='upper')
        
    # 4. Render Perimeter Contour Lines with Explicit Meshgrid Coordinates
    if show_contours and np.any(mask == 1):
        # Create explicit coordinate grid matching (H, W) array indexing
        # X_grid[y, x] = x (horizontal column), Y_grid[y, x] = y (vertical row)
        X_grid, Y_grid = np.meshgrid(np.arange(w), np.arange(h))
        ax.contour(X_grid, Y_grid, mask, levels=[0.5], colors='#FACC15', linewidths=1.8)
        
    ax.set_title(title, color='white', fontsize=12, pad=10, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig

def export_overlay_as_png(raw_image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> bytes:
    """
    Exports high-resolution PNG image buffer of the segmented overlay.
    """
    fig = render_segmentation_figure(raw_image, mask, alpha=alpha, show_overlay=True, show_contours=True, title="")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0, facecolor='#0E1117')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
