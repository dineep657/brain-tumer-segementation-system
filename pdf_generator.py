import os
import datetime
from fpdf import FPDF
from typing import Dict, Any

class TumorReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 58, 138)
        self.cell(0, 10, 'Academic 2D Brain Tumor Segmentation & Classification Report', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'For Academic Demonstration Only | PyTorch UNet & Classifier | Page {self.page_no()}', align='C')

def generate_2d_pdf_report(filename: str, metrics: Dict[str, Any], slice_image_path: str = None) -> bytes:
    """
    Generates a B.Tech project PDF report with real UNet segmentation & PyTorch classifier predictions.
    """
    pdf = TumorReportPDF()
    pdf.add_page()
    
    # Academic Disclaimer Box
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(239, 68, 68)
    pdf.set_text_color(185, 28, 28)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 7, "  RESEARCH DISCLAIMER: For academic demonstration only. Not a clinical diagnostic tool.", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    # 1. Metadata Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Scan & Model Telemetry Metadata", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 6, f"File Name: {filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Segmentation Checkpoint: models/brain_tumor_unet_2d.pth (LightweightUNet2D)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Classifier Checkpoint: models/brain_tumor_classifier_2d.pth (LightweightClassifier2D)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Supported Classes: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    
    # 2. Executive Detection Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Executive Diagnostic & Classification Summary", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 11)
    if metrics["tumor_detected"]:
        pdf.set_text_color(185, 28, 28) # Red
        status_text = f"POSITIVE - TUMOR DETECTED ({metrics['tumor_area_pixels']:,} pixels)"
    else:
        pdf.set_text_color(21, 128, 61) # Green
        status_text = "NEGATIVE - NO TUMOR DETECTED"
        
    pdf.cell(0, 7, f"Diagnosis Status: {status_text}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 7, f"Predicted Tumor Type: {metrics['tumor_subtype_label']} (Classifier Confidence: {metrics['classification_confidence']})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    
    # 3. Quantitative Analytics Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Quantitative Model Segmentation Analytics", new_x="LMARGIN", new_y="NEXT")
    
    col_width = 90
    row_height = 8
    
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_width, row_height, " Metric", border=1, fill=True)
    pdf.cell(col_width, row_height, " Measured Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(col_width, row_height, " Tumor Detection Status", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_detected_label']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Classified Tumor Type", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_subtype_label']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Classifier Model Confidence", border=1)
    pdf.cell(col_width, row_height, f" {metrics['classification_confidence']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Predicted Tumor Area", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_area_pixels']:,} pixels", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Brain Coverage Ratio", border=1)
    pdf.cell(col_width, row_height, f" {metrics['brain_coverage_pct']:.2f} %", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(8)
    
    # 4. Embedded Image
    if slice_image_path and os.path.exists(slice_image_path):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Segmented MRI Scan Visualization Map", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.image(slice_image_path, x=35, w=140)
        
    return bytes(pdf.output())
