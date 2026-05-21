from __future__ import annotations

from io import BytesIO


def generate_report_pdf(report_markdown: str, priority_level: str = "Non renseigné") -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("Installe reportlab pour activer l'export PDF: uv add reportlab") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Rapport médical préliminaire")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Rapport médical préliminaire", styles["Title"]),
        Paragraph(f"Niveau de priorité : {priority_level}", styles["Heading2"]),
        Spacer(1, 12),
    ]

    for line in report_markdown.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading2"]))
        elif line.strip():
            story.append(Paragraph(line, styles["BodyText"]))
            story.append(Spacer(1, 6))

    doc.build(story)
    return buffer.getvalue()
