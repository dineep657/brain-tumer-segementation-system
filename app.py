import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from inference_2d import preprocess_2d, predict_2d
from metrics_2d import analyze_tumor_metrics
from image_processing import render_segmentation_figure, export_overlay_as_png
from pdf_generator import generate_2d_pdf_report

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="AI 2D Brain Tumor Segmentation & Subtype Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    .stMetric {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .stMetric label {
        color: #94A3B8 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .stMetric .metric-value {
        color: #38BDF8 !important;
        font-weight: 700;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    .status-badge-yes {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-no {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-awaiting {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .status-badge-invalid {
        background-color: rgba(234, 179, 8, 0.2);
        color: #FACC15;
        border: 1px solid #EAB308;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }
    .placeholder-card {
        background-color: #1E293B;
        border: 2px dashed #3B82F6;
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Smart 2D Brain Tumor Segmentation & Subtype Classifier")
st.caption("B.Tech Final Project | Dynamic MONAI UNet Segmentation & Multi-Class Subtype Classifier")

# 2. Sidebar Controls
st.sidebar.header("📁 MRI Input Options")

uploaded_file = st.sidebar.file_uploader("Upload 2D MRI (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

sample_glioma_path = os.path.join("data", "sample_glioma.png")
sample_meningioma_path = os.path.join("data", "sample_meningioma.png")
sample_pituitary_path = os.path.join("data", "sample_pituitary.png")
sample_normal_path = os.path.join("data", "sample_normal_mri.png")
sample_bottle_path = os.path.join("data", "sample_non_mri_bottle.png")

sample_choice = st.sidebar.selectbox(
    "Or Select Preset Sample Scan:",
    options=[
        "None (Awaiting File Upload)",
        "Pituitary Brain Tumor Sample (Positive)",
        "Glioma Brain Tumor Sample (Positive)",
        "Meningioma Brain Tumor Sample (Positive)",
        "Normal Healthy MRI Sample (Negative)",
        "Non-Brain Image (Plastic Bottle Test)"
    ],
    index=0
)

# Determine Input Source
image_source = None
display_filename = None

if uploaded_file is not None:
    display_filename = uploaded_file.name
    image_source = uploaded_file
    st.sidebar.success(f"File loaded: {display_filename}")
elif sample_choice == "Pituitary Brain Tumor Sample (Positive)" and os.path.exists(sample_pituitary_path):
    display_filename = "sample_pituitary.png"
    image_source = sample_pituitary_path
    st.sidebar.info("Preset: Pituitary Brain Tumor.")
elif sample_choice == "Glioma Brain Tumor Sample (Positive)" and os.path.exists(sample_glioma_path):
    display_filename = "sample_glioma.png"
    image_source = sample_glioma_path
    st.sidebar.info("Preset: Glioma Brain Tumor.")
elif sample_choice == "Meningioma Brain Tumor Sample (Positive)" and os.path.exists(sample_meningioma_path):
    display_filename = "sample_meningioma.png"
    image_source = sample_meningioma_path
    st.sidebar.info("Preset: Meningioma Brain Tumor.")
elif sample_choice == "Normal Healthy MRI Sample (Negative)" and os.path.exists(sample_normal_path):
    display_filename = "sample_normal_mri.png"
    image_source = sample_normal_path
    st.sidebar.info("Preset: Healthy Normal Brain MRI.")
elif sample_choice == "Non-Brain Image (Plastic Bottle Test)" and os.path.exists(sample_bottle_path):
    display_filename = "sample_non_mri_bottle.png"
    image_source = sample_bottle_path
    st.sidebar.warning("Preset: Non-Brain Image Test.")

# Display Controls
st.sidebar.subheader("👁️ Display & Contrast Controls")
show_overlay = st.sidebar.checkbox("Show Tumor Mask Overlay", value=True)
high_contrast = st.sidebar.checkbox("High Contrast Mode (2%-98% Scaling)", value=False)
show_contours = st.sidebar.checkbox("Highlight Perimeter Contours", value=True)
overlay_alpha = st.sidebar.slider("Overlay Opacity (Alpha)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

# NEUTRAL AWAITING STATE (No file uploaded & No preset selected)
if image_source is None:
    st.subheader("📊 Quantitative Clinical Metrics")
    card_col1, card_col2, card_col3, card_col4 = st.columns(4)
    
    with card_col1:
        st.markdown("**Tumor Detected?**<br><span class='status-badge-awaiting'>ℹ️ AWAITING INPUT</span>", unsafe_allow_html=True)
    with card_col2:
        st.metric(label="Predicted Tumor Subtype", value="-", delta="Awaiting image upload")
    with card_col3:
        st.metric(label="Tumor Area", value="-", delta="Awaiting image upload")
    with card_col4:
        st.metric(label="Model Confidence", value="-", delta="Awaiting image upload")
        
    st.markdown("---")
    
    st.markdown("""
        <div class='placeholder-card'>
            <h2 style='color:#38BDF8;'>📁 Ready for Image Upload</h2>
            <p style='color:#94A3B8; font-size:1.1rem;'>
                Please upload a 2D Brain MRI scan (<strong>.png, .jpg, .jpeg</strong>) using the sidebar file uploader,<br>
                or select a preset sample scan (e.g. <em>Pituitary Tumor, Glioma, Meningioma</em>) to initiate real-time AI segmentation & subtype classification.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Tip:** Selecting a preset scan or uploading a custom MRI will instantly run MONAI UNet inference and dynamically classify the tumor subtype.")
    st.stop()

# Dynamic Inference Execution (Only runs when image_source is provided)
def run_dynamic_pipeline(src):
    tensor, raw_2d = preprocess_2d(src)
    pred_info = predict_2d(tensor, raw_2d)
    mask = pred_info["mask"]
    
    metrics = analyze_tumor_metrics(
        mask, raw_2d,
        confidence=pred_info["confidence"],
        tumor_type=pred_info["tumor_type"],
        type_confidence=pred_info["tumor_type_confidence"]
    )
    
    os.makedirs("data", exist_ok=True)
    overlay_img_path = os.path.join("data", "2d_tumor_overlay.png")
    
    fig_pdf = render_segmentation_figure(
        raw_2d, mask, alpha=0.5, show_overlay=pred_info["is_valid_mri"],
        high_contrast=False, show_contours=pred_info["is_valid_mri"], title="2D MRI Segmentation Result"
    )
    fig_pdf.savefig(overlay_img_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig_pdf)
    
    png_bytes = export_overlay_as_png(raw_2d, mask, alpha=0.5)
    
    return raw_2d, mask, metrics, pred_info, overlay_img_path, png_bytes

try:
    raw_2d, mask, metrics, pred_info, overlay_img_path, png_bytes = run_dynamic_pipeline(image_source)
except Exception as e:
    st.error(f"Error processing image: {e}")
    st.stop()

# Handle Invalid Non-MRI Input Warning Banner
if not pred_info["is_valid_mri"]:
    st.error(f"🚨 {pred_info['validation_error']}")

# 3. Sidebar Exports
st.sidebar.subheader("📥 Export & Report Options")

try:
    pdf_bytes = generate_2d_pdf_report(
        filename=display_filename,
        metrics=metrics,
        slice_image_path=overlay_img_path
    )
    
    st.sidebar.download_button(
        label="📄 Download PDF Summary Report",
        data=pdf_bytes,
        file_name=f"Brain_Tumor_Report_{display_filename}.pdf",
        mime="application/pdf"
    )
except Exception as pdf_err:
    st.sidebar.error(f"Unable to generate PDF report: {pdf_err}")

st.sidebar.download_button(
    label="🖼️ Download Segmented Image (.png)",
    data=png_bytes,
    file_name=f"Segmented_{display_filename}.png",
    mime="image/png"
)

# 4. Active Results Metrics Cards
st.subheader("📊 Quantitative Clinical Metrics")

card_col1, card_col2, card_col3, card_col4 = st.columns(4)

with card_col1:
    if not pred_info["is_valid_mri"]:
        status_html = "<span class='status-badge-invalid'>⚠️ INVALID INPUT</span>"
    elif metrics["tumor_detected"]:
        status_html = "<span class='status-badge-yes'>🔴 DETECTED</span>"
    else:
        status_html = "<span class='status-badge-no'>🟢 NO TUMOR DETECTED</span>"
        
    st.markdown(f"**Tumor Detected?**<br>{status_html}", unsafe_allow_html=True)

with card_col2:
    st.metric(
        label="Predicted Tumor Subtype",
        value=metrics["tumor_type"],
        delta=f"{metrics['tumor_type_confidence_str']} Subtype Confidence"
    )

with card_col3:
    st.metric(
        label="Tumor Area",
        value=f"{metrics['tumor_area_pixels']:,} Pixels",
        delta="Calculated from segmentation mask"
    )

with card_col4:
    st.metric(
        label="Model Confidence",
        value=metrics['confidence_score'],
        delta="Mean softmax probability"
    )

st.markdown("---")

# 5. Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📷 Input Image Slice")
    fig_raw = render_segmentation_figure(
        raw_2d, mask, show_overlay=False, high_contrast=high_contrast,
        title="Input Image Grayscale Matrix (256x256)"
    )
    st.pyplot(fig_raw, width="stretch")
    plt.close(fig_raw)

with col_right:
    st.subheader("🎯 MONAI UNet Segmentation Overlay")
    fig_segmented = render_segmentation_figure(
        raw_2d, mask, alpha=overlay_alpha, show_overlay=show_overlay and pred_info["is_valid_mri"],
        high_contrast=high_contrast, show_contours=show_contours and pred_info["is_valid_mri"],
        title=f"Subtype: {pred_info['tumor_type']} Overlay" if pred_info["is_valid_mri"] else "Segmentation Skipped (Invalid MRI)"
    )
    st.pyplot(fig_segmented, width="stretch")
    plt.close(fig_segmented)

# 6. Viva Panel: Model Diagnostics & Subtype Classification Telemetry
st.markdown("---")
with st.expander("🔬 Model Inference Diagnostics & Telemetry (Viva Presentation Panel)", expanded=True):
    v_col1, v_col2, v_col3 = st.columns(3)
    
    with v_col1:
        st.markdown(f"**Model Load Status:**<br>`{pred_info['model_status']}`", unsafe_allow_html=True)
        st.markdown(f"**PyTorch Forward Pass:**<br>`{'EXECUTED' if pred_info['model_called'] else 'SKIPPED'}`", unsafe_allow_html=True)
        
    with v_col2:
        st.markdown(f"**Input Tensor Shape:**<br>`{pred_info['tensor_shape']}`", unsafe_allow_html=True)
        st.markdown(f"**Inference Execution Time:**<br>`⚡ {pred_info['execution_time_ms']} ms`", unsafe_allow_html=True)
        
    with v_col3:
        st.markdown(f"**Predicted Subtype:**<br>`{pred_info['tumor_type']} ({pred_info['tumor_type_confidence']*100:.1f}%)`", unsafe_allow_html=True)
        st.markdown(f"**Segmented Tumor Pixels:**<br>`{pred_info['tumor_pixel_count']} voxels`", unsafe_allow_html=True)
