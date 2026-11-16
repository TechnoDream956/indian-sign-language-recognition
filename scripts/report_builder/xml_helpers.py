"""XML and python-docx helpers for building the ISL College Project Report.
Handles field codes, table styling, formatting, and yellow highlight placeholders.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def enable_update_fields_on_open(doc: Document):
    """Ensure Word updates TOC, LOT, and LOF fields automatically when opened."""
    settings = doc.settings.element
    update_fields = settings.find(qn('w:updateFields'))
    if update_fields is None:
        update_fields = parse_xml(f'<w:updateFields {nsdecls("w")} w:val="true"/>')
        settings.append(update_fields)
    else:
        update_fields.set(qn('w:val'), 'true')


def add_styled_paragraph(doc: Document, text: str, style_name: str = 'Body Double Space 0.5 First Line', 
                         align=None, space_before=None, space_after=None, line_spacing=None):
    """Add a paragraph with explicit style and formatting."""
    p = doc.add_paragraph(style=style_name)
    if text:
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
    if align is not None:
        p.alignment = align
    if space_before is not None:
        p.paragraph_format.space_before = space_before
    if space_after is not None:
        p.paragraph_format.space_after = space_after
    if line_spacing is not None:
        p.paragraph_format.line_spacing = line_spacing
    return p


def add_body_p(doc: Document, text: str):
    """Add standard body paragraph with Times New Roman 12pt, justified, double spaced."""
    return add_styled_paragraph(doc, text, style_name='Body Double Space 0.5 First Line', 
                                align=WD_ALIGN_PARAGRAPH.JUSTIFY)


def add_bullet_p(doc: Document, text: str, bold_prefix: str = ""):
    """Add a bullet point paragraph matching template Bullets style."""
    p = doc.add_paragraph(style='Bullets')
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.bold = True
    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    return p


def add_todo_placeholder(p, todo_text: str):
    """Add a yellow-highlighted [TODO: ...] placeholder to a paragraph."""
    run = p.add_run(f" [TODO: {todo_text}]")
    run.font.name = 'Times New Roman'
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.bold = True
    return run


def add_heading_1(doc: Document, text: str):
    """Add a Chapter level heading (GS1)."""
    p = doc.add_paragraph(style='GS1')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.bold = True
    return p


def add_heading_2(doc: Document, text: str):
    """Add a Section level heading (GS2)."""
    p = doc.add_paragraph(style='GS2')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.bold = True
    return p


def add_heading_3(doc: Document, text: str):
    """Add a Subsection level heading (GS3)."""
    p = doc.add_paragraph(style='GS3')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = True
    return p


def add_table_caption(doc: Document, table_num: int, title: str):
    """Add a standard Word Table Caption using SEQ Table field."""
    p = doc.add_paragraph(style='Caption')
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0

    r1 = p.add_run("Table ")
    r1.font.name = 'Times New Roman'
    r1.bold = True

    # Insert SEQ Table field
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'SEQ Table \\* ARABIC')
    # Default text for viewers that don't evaluate fields immediately
    r_val = OxmlElement('w:r')
    r_val_t = OxmlElement('w:t')
    r_val_t.text = str(table_num)
    r_val.append(r_val_t)
    fldSimple.append(r_val)
    p._p.append(fldSimple)

    r2 = p.add_run(f": {title}")
    r2.font.name = 'Times New Roman'
    r2.bold = True
    return p


def add_figure_caption(doc: Document, fig_num: int, title: str):
    """Add a standard Word Figure Caption using SEQ Figure field."""
    p = doc.add_paragraph(style='Caption')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(12)
    p.paragraph_format.line_spacing = 1.0

    r1 = p.add_run("Figure ")
    r1.font.name = 'Times New Roman'
    r1.bold = True

    # Insert SEQ Figure field
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'SEQ Figure \\* ARABIC')
    r_val = OxmlElement('w:r')
    r_val_t = OxmlElement('w:t')
    r_val_t.text = str(fig_num)
    r_val.append(r_val_t)
    fldSimple.append(r_val)
    p._p.append(fldSimple)

    r2 = p.add_run(f": {title}")
    r2.font.name = 'Times New Roman'
    r2.bold = True
    return p


def insert_image_centered(doc: Document, image_path: str, width_inches: float = 6.0):
    """Insert an image centered on the page."""
    if not os.path.exists(image_path):
        p_err = doc.add_paragraph(style='Normal')
        add_todo_placeholder(p_err, f"Missing image at {os.path.basename(image_path)}")
        return None
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run()
    run.add_picture(image_path, width=Inches(width_inches))
    return p


def style_table_cell(cell, bg_color=None, bold=False, italic=False, font_size=10, 
                     align=WD_ALIGN_PARAGRAPH.LEFT, text_color="000000"):
    """Format an individual table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    if bg_color:
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_color}"/>')
        tcPr.append(shd)
    
    # Cell padding / margins
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="120" w:type="dxa"/><w:bottom w:w="120" w:type="dxa"/><w:left w:w="160" w:type="dxa"/><w:right w:w="160" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    for p in cell.paragraphs:
        p.alignment = align
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        for r in p.runs:
            r.font.name = 'Times New Roman'
            r.font.size = Pt(font_size)
            r.bold = bold
            r.italic = italic
            if text_color != "000000":
                r.font.color.rgb = RGBColor.from_string(text_color)


def apply_table_borders(table):
    """Apply clean academic borders to a Word table (top, bottom, header rule)."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="single" w:sz="8" w:space="0" w:color="333333"/>
            <w:bottom w:val="single" w:sz="8" w:space="0" w:color="333333"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
            <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)


def format_academic_table(table, col_widths=None, header_bg="1F497D", alt_bg="F2F5F9"):
    """Format an entire table with academic styling."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    apply_table_borders(table)

    # Format header row
    header_tr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = parse_xml(f'<w:tblHeader {nsdecls("w")}/>')
    header_tr.append(tblHeader)

    for cell in table.rows[0].cells:
        style_table_cell(cell, bg_color=header_bg, bold=True, font_size=10, 
                         align=WD_ALIGN_PARAGRAPH.CENTER, text_color="FFFFFF")

    # Format body rows
    for r_idx, row in enumerate(table.rows[1:], start=1):
        bg = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            style_table_cell(cell, bg_color=bg, bold=False, font_size=9.5, 
                             align=WD_ALIGN_PARAGRAPH.LEFT)

    # Column widths
    if col_widths:
        for row in table.rows:
            for idx, w in enumerate(col_widths):
                if idx < len(row.cells):
                    row.cells[idx].width = Inches(w)
