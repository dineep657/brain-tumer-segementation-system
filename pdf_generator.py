import os
import datetime
from fpdf import FPDF
from typing import Dict, Any

class TumorReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 58, 138)  # Deep Navy Blue
        self.cell(0, 10, 'AI-Powered 2D Brain Tumor Analysis Report', border=False, new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.5)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'B.Tech Final Project | MONAI UNet & Multi-Class Subtype Classifier | Page {self.page_no()}', align='C')

def generate_2d_pdf_report(filename: str, metrics: Dict[str, Any], slice_image_path: str = None) -> bytes:
    """
    Generates a B.Tech project clinical PDF summary report containing metadata,
    executive diagnosis status, single tumor subtype, quantitative metrics table, and embedded overlay image.
    """
    pdf = TumorReportPDF()
    pdf.add_page()
    
    # 1. Metadata Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Scan & System Metadata", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 6, f"File Name: {filename}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Deep Learning Architecture: MONAI 2D UNet & Multi-Class Subtype Classifier", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Matrix Resolution: 256 x 256 pixels", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    
    # 2. Executive Detection & Classification Summary
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Executive Clinical Summary", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "B", 11)
    if metrics["tumor_detected"]:
        pdf.set_text_color(185, 28, 28) # Red
        status_text = f"POSITIVE - TUMOR DETECTED: {metrics['tumor_type']} ({metrics['tumor_type_confidence_str']})"
    else:
        pdf.set_text_color(21, 128, 61) # Green
        status_text = "NEGATIVE - HEALTHY BRAIN (NO TUMOR DETECTED)"
        
    pdf.cell(0, 7, f"Diagnosis Status: {status_text}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(51, 65, 85)
    pdf.ln(4)
    
    # 3. Quantitative Analytics Table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Quantitative Segmentation & Subtype Analytics", new_x="LMARGIN", new_y="NEXT")
    
    col_width = 90
    row_height = 8
    
    pdf.set_fill_color(226, 232, 240)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(col_width, row_height, " Clinical / Technical Metric", border=1, fill=True)
    pdf.cell(col_width, row_height, " Measured Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(col_width, row_height, " Tumor Detection Status", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_detected_label']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Predicted Tumor Subtype", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_type']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Subtype Classification Confidence", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_type_confidence_str']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Tumor Area (Pixels)", border=1)
    pdf.cell(col_width, row_height, f" {metrics['tumor_area_pixels']:,} pixels", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Brain Coverage Ratio", border=1)
    pdf.cell(col_width, row_height, f" {metrics['brain_coverage_pct']:.2f} %", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(col_width, row_height, " Model Prediction Confidence", border=1)
    pdf.cell(col_width, row_height, f" {metrics['confidence_score']}", border=1, new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(8)
    
    # 4. Embedded Image
    if slice_image_path and os.path.exists(slice_image_path):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "Segmented MRI Scan Visualization", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.image(slice_image_path, x=35, w=140)
        
    return bytes(pdf.output())

if __name__ == "__main__":
    test_metrics = {
        "tumor_detected": True,
        "tumor_detected_label": "YES",
        "tumor_type": "Pituitary Tumor",
        "tumor_type_confidence_str": "94.2%",
        "tumor_area_pixels": 2826,
        "brain_area_pixels": 31415,
        "brain_coverage_pct": 9.00,
        "confidence_score": "94.5%",
        "raw_confidence": 0.945
    }
    test_pdf = generate_2d_pdf_report("sample_pituitary.png", test_metrics)
    print(f"Single Subtype PDF Report generated: {len(test_pdf)} bytes")
