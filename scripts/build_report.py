#!/usr/bin/env python3
"""Master build script for ISL College Project Report.
Builds docs/report/ISL_Alphabet_Recognition_Project_Report.docx from the official Bennett template.
"""

import os
import sys
import zipfile
from docx import Document

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.report_builder.diagram_generator import generate_all_diagrams
from scripts.report_builder.xml_helpers import enable_update_fields_on_open
from scripts.report_builder.cover_page import build_cover_page
from scripts.report_builder.front_matter import build_front_matter
from scripts.report_builder.chapter1_introduction import build_chapter1
from scripts.report_builder.chapter2_background import build_chapter2
from scripts.report_builder.chapter3_methodology import build_chapter3
from scripts.report_builder.chapter4_conclusion import build_chapter4
from scripts.report_builder.chapter5_references import build_chapter5
from scripts.report_builder.todo_extractor import extract_todos

TEMPLATE_PATH = os.path.join(REPO_ROOT, "docs/report/AI_ML_Project_Report_Template.docx")
OUTPUT_PATH = os.path.join(REPO_ROOT, "docs/report/ISL_Alphabet_Recognition_Project_Report.docx")


def strip_sample_body(doc: Document):
    """Safely remove template sample body paragraphs after Section 1 break (P141) and sample table."""
    body = doc._body._element
    elements_to_remove = []
    p_count = 0
    for child in list(body):
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            if p_count > 141:
                elements_to_remove.append(child)
            p_count += 1
        elif tag == 'tbl':
            elements_to_remove.append(child)

    for elem in elements_to_remove:
        body.remove(elem)

    print(f"Removed {len(elements_to_remove)} template placeholder body elements.")


def verify_xml_cleanliness(output_docx_path: str):
    """Verify that no <<...>> placeholder brackets exist in document.xml."""
    with zipfile.ZipFile(output_docx_path, 'r') as z:
        xml_content = z.read('word/document.xml').decode('utf-8')
    
    count_open = xml_content.count('&lt;&lt;') + xml_content.count('<<')
    count_close = xml_content.count('&gt;&gt;') + xml_content.count('>>')
    
    print(f"XML Validation: Found {count_open} template opening brackets ('<<') and {count_close} closing brackets ('>>').")
    if count_open > 0 or count_close > 0:
        print("WARNING: Residual template instruction brackets detected!")
    else:
        print("SUCCESS: 100% of template <<...>> instruction brackets have been stripped.")


def main():
    print("=" * 70)
    print("Starting Indian Sign Language Project Report Generator")
    print("=" * 70)

    # 1. Generate Figures
    print("\n[Step 1/6] Generating Publication-Quality Figures & Diagrams...")
    generate_all_diagrams()

    # 2. Load Template
    print(f"\n[Step 2/6] Loading Official Bennett Template: {TEMPLATE_PATH}")
    if not os.path.exists(TEMPLATE_PATH):
        print(f"ERROR: Template not found at {TEMPLATE_PATH}")
        sys.exit(1)
    
    doc = Document(TEMPLATE_PATH)
    print(f"Loaded template with {len(doc.sections)} sections and {len(doc.paragraphs)} initial paragraphs.")

    # 3. Build Cover Page & Front Matter in-place
    print("\n[Step 3/6] Updating Cover Page & Front Matter In-Place...")
    build_cover_page(doc)
    build_front_matter(doc)

    # 4. Remove Template Sample Body
    print("\n[Step 4/6] Stripping Template Instructions & Sample Tables...")
    strip_sample_body(doc)

    # 5. Build Academic Chapters (1 to 5)
    print("\n[Step 5/6] Appending Chapters 1 through 5...")
    build_chapter1(doc)
    build_chapter2(doc)
    build_chapter3(doc)
    build_chapter4(doc)
    build_chapter5(doc)

    # 6. Configure Settings, Save & Audit
    print("\n[Step 6/6] Finalizing Document Settings & Saving Output...")
    enable_update_fields_on_open(doc)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    doc.save(OUTPUT_PATH)
    print(f"\nSaved final project report to: {OUTPUT_PATH}")
    print(f"Final paragraph count: {len(doc.paragraphs)}")
    print(f"Final table count: {len(doc.tables)}")
    print(f"Final section count: {len(doc.sections)}")

    # Audit TODOs
    extract_todos(doc)

    # Verify XML cleanliness
    verify_xml_cleanliness(OUTPUT_PATH)

    print("\n" + "=" * 70)
    print("Report Generation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
