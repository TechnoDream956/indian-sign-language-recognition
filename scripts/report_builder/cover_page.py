"""Cover page builder for ISL Project Report.
Updates the cover page in-place while preserving the Bennett University logo drawing.
"""

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def build_cover_page(doc: Document):
    """Update cover page in-place preserving drawing at paragraph 9."""
    
    # Title
    p0 = doc.paragraphs[0]
    p0.text = "REAL-TIME INDIAN SIGN LANGUAGE ALPHABET RECOGNITION WITH MULTILINGUAL TRANSLATION AND SPEECH OUTPUT"
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p0.runs:
        p0.runs[0].font.name = 'Times New Roman'
        p0.runs[0].font.size = Pt(16)
        p0.runs[0].bold = True

    # Course info
    p1 = doc.paragraphs[1]
    p1.text = "A MINI-PROJECT REPORT (CSET312)"
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p1.runs:
        p1.runs[0].font.name = 'Times New Roman'
        p1.runs[0].font.size = Pt(13)
        p1.runs[0].bold = True

    # BY
    p2 = doc.paragraphs[2]
    p2.text = "BY"
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p2.runs:
        p2.runs[0].font.name = 'Times New Roman'
        p2.runs[0].font.size = Pt(12)
        p2.runs[0].bold = True

    # Student Names & Enrollment (4 members) - Preserved as TODO placeholders
    students = [
        "ARPIT [TODO: Full Name] (Enrollment No. [TODO: Enrollment No.])",
        "SACHIN [TODO: Full Name] (Enrollment No. [TODO: Enrollment No.])",
        "ABHU [TODO: Full Name] (Enrollment No. [TODO: Enrollment No.])",
        "YASH [TODO: Full Name] (Enrollment No. [TODO: Enrollment No.])"
    ]

    p5 = doc.paragraphs[5]
    p5.text = students[0]
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p5.runs:
        p5.runs[0].font.name = 'Times New Roman'
        p5.runs[0].font.size = Pt(12)
        p5.runs[0].bold = True

    p6 = doc.paragraphs[6]
    p6.text = students[1]
    p6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p6.runs:
        p6.runs[0].font.name = 'Times New Roman'
        p6.runs[0].font.size = Pt(12)
        p6.runs[0].bold = True

    p7 = doc.paragraphs[7]
    p7.text = students[2]
    p7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p7.runs:
        p7.runs[0].font.name = 'Times New Roman'
        p7.runs[0].font.size = Pt(12)
        p7.runs[0].bold = True

    p8 = doc.paragraphs[8]
    p8.text = students[3]
    p8.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p8.runs:
        p8.runs[0].font.name = 'Times New Roman'
        p8.runs[0].font.size = Pt(12)
        p8.runs[0].bold = True

    # Paragraph 9 contains the Bennett University logo drawing - DO NOT TOUCH

    # Submission footer
    p14 = doc.paragraphs[14]
    p14.text = "SUBMITTED TO"
    p14.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p14.runs:
        p14.runs[0].font.name = 'Times New Roman'
        p14.runs[0].font.size = Pt(12)
        p14.runs[0].bold = True

    p16 = doc.paragraphs[16]
    p16.text = "SCHOOL OF COMPUTER SCIENCE ENGINEERING AND TECHNOLOGY, BENNETT UNIVERSITY"
    p16.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p16.runs:
        p16.runs[0].font.name = 'Times New Roman'
        p16.runs[0].font.size = Pt(12)
        p16.runs[0].bold = True

    p17 = doc.paragraphs[17]
    p17.text = "GREATER NOIDA, 201310, UTTAR PRADESH, INDIA"
    p17.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p17.runs:
        p17.runs[0].font.name = 'Times New Roman'
        p17.runs[0].font.size = Pt(12)
        p17.runs[0].bold = True

    p19 = doc.paragraphs[19]
    p19.text = "\t[TODO: Month, Year]"
    p19.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if p19.runs:
        p19.runs[0].font.name = 'Times New Roman'
        p19.runs[0].font.size = Pt(12)
        p19.runs[0].bold = True

    print("Cover page updated successfully.")
