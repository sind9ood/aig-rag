"""Document-aware line classification with page/footer context tracking.

This module owns:
- footer parsing and page-context extraction
- line labeling (table_row / narrative / footer)
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.indexing.table_row_detection import predict_line_label


class LineLabel(Enum):
    TABLE_ROW = "table_row"
    NARRATIVE = "narrative"
    FOOTER = "footer"


@dataclass
class PageContext:
    page_number: Optional[int] = None
    form_label: Optional[str] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    subsubsection: Optional[str] = None

    def to_header_str(self):
        parts = []
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.form_label:
            parts.append(self.form_label)
        if self.section:
            parts.append(self.section)
        if self.subsection:
            parts.append(self.subsection)
        if self.subsubsection:
            parts.append(self.subsubsection)
        return " | ".join(parts) if parts else ""


@dataclass
class FooterParseResult:
    previous_page_footer: PageContext
    following_page_footnote: PageContext
    lines_consumed: int


@dataclass
class LineClassification:
    line_index: int
    line: str
    label: LineLabel
    page_ctx: PageContext


PAGE_NUMBER_RE = re.compile(r"^\s*(\d{1,3})\s*$")
FORM_LABEL_RE = re.compile(r"(AIG\s*\|\s*\d{4}\s*Form\s*10-[KQ])", re.IGNORECASE)
FOOTER_SECTION_RE = re.compile(
    r"^(ITEM\s*\d+[A-Z]?)\s*\|\s*([^|]+?)(?:\|\s*(.+))?$", re.IGNORECASE
)


def parse_footer_block(lines, start_idx):
    prev_ctx = PageContext()
    next_ctx = PageContext()

    i = start_idx
    consumed = 0
    found_any = False

    def _consume_blanks(idx):
        nonlocal consumed
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
            consumed += 1
        return idx

    i = _consume_blanks(i)
    if i >= len(lines):
        return None

    first = lines[i].strip()
    if not (
        PAGE_NUMBER_RE.match(first)
        or FORM_LABEL_RE.search(first)
        or FOOTER_SECTION_RE.match(first)
    ):
        return None

    max_nonblank_footer_lines = 3
    nonblank_parsed = 0
    while i < len(lines) and nonblank_parsed < max_nonblank_footer_lines:
        i = _consume_blanks(i)
        if i >= len(lines):
            break

        s = lines[i].strip()
        matched = False

        m_page = PAGE_NUMBER_RE.match(s)
        if m_page and prev_ctx.page_number is None:
            prev_ctx.page_number = int(m_page.group(1))
            matched = True
        elif FORM_LABEL_RE.search(s) and prev_ctx.form_label is None:
            prev_ctx.form_label = s
            matched = True
        else:
            m_section = FOOTER_SECTION_RE.match(s)
            if m_section and next_ctx.section is None:
                next_ctx.section = m_section.group(1).strip()
                next_ctx.subsection = m_section.group(2).strip() if m_section.group(2) else None
                next_ctx.subsubsection = m_section.group(3).strip() if m_section.group(3) else None
                matched = True

        if not matched:
            break

        found_any = True
        nonblank_parsed += 1
        i += 1
        consumed += 1

    if not found_any:
        return None

    return FooterParseResult(
        previous_page_footer=prev_ctx,
        following_page_footnote=next_ctx,
        lines_consumed=consumed,
    )


def classify_lines(text):
    raw_lines = text.replace("\xa0", " ").split("\n")
    results = []

    current_page = PageContext()

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        footer_result = parse_footer_block(raw_lines, i)
        if footer_result and footer_result.lines_consumed > 0:
            for j in range(i, i + footer_result.lines_consumed):
                footer_line = raw_lines[j]
                if not footer_line.strip():
                    continue
                results.append(
                    LineClassification(
                        line_index=j,
                        line=footer_line,
                        label=LineLabel.FOOTER,
                        page_ctx=PageContext(
                            page_number=current_page.page_number,
                            form_label=current_page.form_label,
                            section=current_page.section,
                            subsection=current_page.subsection,
                            subsubsection=current_page.subsubsection,
                        ),
                    )
                )

            prev_footer = footer_result.previous_page_footer
            next_footnote = footer_result.following_page_footnote

            if prev_footer.page_number is not None:
                current_page.page_number = prev_footer.page_number
            if prev_footer.form_label:
                current_page.form_label = prev_footer.form_label

            if next_footnote.section:
                current_page.section = next_footnote.section
                current_page.subsection = next_footnote.subsection
                current_page.subsubsection = next_footnote.subsubsection

            i += footer_result.lines_consumed
            continue

        if stripped:
            pred = predict_line_label(raw_lines, i)
            label = LineLabel.TABLE_ROW if pred == "table_row" else LineLabel.NARRATIVE
            results.append(
                LineClassification(
                    line_index=i,
                    line=line,
                    label=label,
                    page_ctx=PageContext(
                        page_number=current_page.page_number,
                        form_label=current_page.form_label,
                        section=current_page.section,
                        subsection=current_page.subsection,
                        subsubsection=current_page.subsubsection,
                    ),
                )
            )

        i += 1

    return results


__all__ = [
    "LineLabel",
    "PageContext",
    "FooterParseResult",
    "LineClassification",
    "parse_footer_block",
    "classify_lines",
]
