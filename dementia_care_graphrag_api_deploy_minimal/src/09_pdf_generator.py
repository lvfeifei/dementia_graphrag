"""
Step 09: PDF care plan generator.

This tool reads Step 06 answer result and generates a PDF care plan.

Input by default:
- outputs/answers/last_answer_result.json

Outputs:
- outputs/pdfs/last_care_plan.pdf
- outputs/pdfs/dementia_care_plan_<timestamp>.pdf

This script does NOT modify Neo4j.
It only reads local answer JSON and writes PDF files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANSWERS_DIR = PROJECT_ROOT / "outputs" / "answers"
PDF_DIR = PROJECT_ROOT / "outputs" / "pdfs"

DEFAULT_ANSWER_RESULT_PATH = ANSWERS_DIR / "last_answer_result.json"
LAST_PDF_PATH = PDF_DIR / "last_care_plan.pdf"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def load_answer_result(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Answer result not found: {path}\n"
            "Please run Step 06 or Step 08 first."
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Answer result should be a JSON object: {path}")

    return data


def clean_markdown_line(line: str) -> str:
    line = line.strip()

    # Remove markdown heading symbols.
    line = re.sub(r"^#{1,6}\s*", "", line)

    # Remove bold/italic markers.
    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
    line = re.sub(r"\*(.*?)\*", r"\1", line)

    # Normalize bullets.
    line = re.sub(r"^\s*[-*]\s+", "• ", line)

    return line.strip()


def is_section_heading(line: str) -> bool:
    line = clean_markdown_line(line)
    return bool(re.match(r"^[一二三四五六七八九十]+、", line))


def split_answer_lines(answer_md: str) -> List[str]:
    lines: List[str] = []
    for raw in answer_md.splitlines():
        line = raw.rstrip()
        lines.append(line)
    return lines


def selected_problem_text(answer_result: Dict[str, Any]) -> List[str]:
    problems = answer_result.get("selected_problems") or []
    lines: List[str] = []

    for p in problems:
        pid = clean_text(p.get("id"))
        category = clean_text(p.get("category"))
        content = clean_text(p.get("content") or p.get("name"))
        confidence = p.get("confidence")

        line = f"{pid}｜{category}｜{content}"
        if confidence is not None:
            line += f"｜置信度：{confidence}"
        lines.append(line)

    if not lines:
        ids = answer_result.get("selected_problem_ids") or []
        if ids:
            lines.append("、".join(ids))

    return lines


def evidence_source_text(answer_result: Dict[str, Any]) -> List[str]:
    """
    Try to extract source lines from prompt_context.
    """
    context = answer_result.get("prompt_context") or {}
    source_lines = context.get("evidence_source_lines") or []

    cleaned: List[str] = []

    for line in source_lines:
        line = clean_text(line)
        if line:
            cleaned.append(line)

    return cleaned


def import_reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.pdfbase import pdfmetrics
        from reportlab.lib import colors
    except ImportError as e:
        raise ImportError(
            "reportlab is not installed.\n"
            "Please run:\n"
            "  pip install reportlab"
        ) from e

    return {
        "A4": A4,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle,
        "TA_LEFT": TA_LEFT,
        "TA_CENTER": TA_CENTER,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "PageBreak": PageBreak,
        "UnicodeCIDFont": UnicodeCIDFont,
        "pdfmetrics": pdfmetrics,
        "colors": colors,
    }


def generate_pdf(
    answer_result: Dict[str, Any],
    output_path: Path,
) -> Path:
    rl = import_reportlab()

    A4 = rl["A4"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    ParagraphStyle = rl["ParagraphStyle"]
    TA_LEFT = rl["TA_LEFT"]
    TA_CENTER = rl["TA_CENTER"]
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    UnicodeCIDFont = rl["UnicodeCIDFont"]
    pdfmetrics = rl["pdfmetrics"]
    colors = rl["colors"]

    ensure_dirs()

    # Built-in CJK font in ReportLab.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=50,
        bottomMargin=42,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="ChineseTitle",
        parent=styles["Title"],
        fontName="STSong-Light",
        fontSize=18,
        leading=26,
        alignment=TA_CENTER,
        spaceAfter=16,
    )

    subtitle_style = ParagraphStyle(
        name="ChineseSubtitle",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=9.5,
        leading=15,
        textColor=colors.HexColor("#666666"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    heading_style = ParagraphStyle(
        name="ChineseHeading",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=13,
        leading=20,
        textColor=colors.HexColor("#222222"),
        spaceBefore=12,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        name="ChineseBody",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=18,
        alignment=TA_LEFT,
        spaceAfter=6,
    )

    small_style = ParagraphStyle(
        name="ChineseSmall",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=9,
        leading=15,
        textColor=colors.HexColor("#666666"),
        spaceAfter=5,
    )

    warning_style = ParagraphStyle(
        name="ChineseWarning",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=9.5,
        leading=16,
        textColor=colors.HexColor("#444444"),
        spaceBefore=10,
        spaceAfter=6,
    )

    story = []

    created_at = clean_text(answer_result.get("created_at"))
    question = clean_text(answer_result.get("question"))
    answer_md = clean_text(answer_result.get("answer_md"))

    story.append(Paragraph("失智老人居家照护建议单", title_style))
    story.append(Paragraph(f"生成时间：{escape(created_at)}", subtitle_style))

    story.append(Paragraph("照护者问题", heading_style))
    story.append(Paragraph(escape(question), body_style))

    problem_lines = selected_problem_text(answer_result)
    story.append(Paragraph("系统识别到的照护问题", heading_style))

    if problem_lines:
        for line in problem_lines:
            story.append(Paragraph("• " + escape(line), body_style))
    else:
        story.append(Paragraph("暂未识别到明确照护问题。", body_style))

    story.append(Spacer(1, 8))
    story.append(Paragraph("照护建议", heading_style))

    for raw_line in split_answer_lines(answer_md):
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        cleaned = clean_markdown_line(line)

        if not cleaned:
            continue

        if is_section_heading(cleaned):
            story.append(Paragraph(escape(cleaned), heading_style))
        else:
            story.append(Paragraph(escape(cleaned), body_style))

    sources = evidence_source_text(answer_result)

    if sources:
        story.append(Spacer(1, 8))
        story.append(Paragraph("主要证据来源", heading_style))

        for line in sources[:12]:
            story.append(Paragraph("• " + escape(line), small_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("安全说明", heading_style))
    story.append(
        Paragraph(
            escape(
                "本建议单用于居家照护教育与照护支持，不能替代医生、护士、记忆门诊或急诊的专业判断。"
                "如老人出现噎食、严重呛咳、意识改变、跌倒后明显疼痛或活动受限、误服后不适、"
                "明显自伤或伤人风险、持续严重幻觉妄想或攻击行为，应及时就医或寻求急诊帮助。"
            ),
            warning_style,
        )
    )

    doc.build(story)

    return output_path


def generate_pdf_from_answer_result(answer_result_path: Path = DEFAULT_ANSWER_RESULT_PATH) -> Dict[str, str]:
    ensure_dirs()

    answer_result = load_answer_result(answer_result_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_pdf_path = PDF_DIR / f"dementia_care_plan_{timestamp}.pdf"

    generate_pdf(answer_result=answer_result, output_path=timestamp_pdf_path)

    shutil.copy2(timestamp_pdf_path, LAST_PDF_PATH)

    return {
        "timestamp_pdf_path": str(timestamp_pdf_path),
        "last_pdf_path": str(LAST_PDF_PATH),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 09: Generate PDF care plan.")
    parser.add_argument(
        "--answer_result",
        type=str,
        default="",
        help="Path to answer result JSON. Default: outputs/answers/last_answer_result.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    answer_result_path = Path(args.answer_result) if args.answer_result else DEFAULT_ANSWER_RESULT_PATH
    if not answer_result_path.is_absolute():
        answer_result_path = PROJECT_ROOT / answer_result_path

    print("=" * 80)
    print("Step 09: PDF care plan generator")
    print("=" * 80)
    print(f"Using answer result: {answer_result_path}")

    paths = generate_pdf_from_answer_result(answer_result_path)

    print("\n[PASS] PDF generated.")
    print(f"[OK] Timestamp PDF: {paths['timestamp_pdf_path']}")
    print(f"[OK] Last PDF: {paths['last_pdf_path']}")


if __name__ == "__main__":
    main()
