"""Document-aware line classification with page/footer context tracking.

This module owns:
- line labeling (table_row / narrative / footer)
- page context tracking from footer lines
"""

import re
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
from src.indexing.train.train_table_detector import ThreeWayEmbedder, build_context_parts

MODEL_PATH = Path("model/table_row_classifier.joblib")


PAGE_NUMBER_RE = re.compile(r"^\s*(\d{1,3})\s*$")
FORM_LABEL_RE = re.compile(r"(AIG\s*\|\s*\d{4}\s*Form\s*10-[KQ])", re.IGNORECASE)
FOOTER_SECTION_RE = re.compile(
    r"^(ITEM\s*\d+[A-Z]?)\s*\|\s*([^|]+?)(?:\|\s*(.+))?$", re.IGNORECASE
)


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
class LineClassification:
    line_index: int
    line: str
    label: LineLabel
    page_ctx: PageContext


def _build_input(lines, idx, window_size, max_line_chars):
    return build_context_parts(lines, idx, window_size, max_line_chars)


@lru_cache(maxsize=1)
def _load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Table detector model not found at {MODEL_PATH}. "
            "Run src/indexing/train/train_table_detector.py first."
        )

    # Preserve backwards compatibility for pickles saved with __main__.ThreeWayEmbedder
    for module_name in ("pytest.__main__", "__main__"):
        module = sys.modules.get(module_name)
        if module is not None and not hasattr(module, "ThreeWayEmbedder"):
            setattr(module, "ThreeWayEmbedder", ThreeWayEmbedder)

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in {MODEL_PATH}")
    return bundle


def predict_line_label(lines, idx):
    bundle = _load_model_bundle()
    model = bundle["model"]
    window_size = int(bundle.get("window_size", 2))
    max_line_chars = int(bundle.get("max_line_chars", 300))
    model_input = _build_input(lines, idx, window_size, max_line_chars)
    pred = model.predict([model_input])[0]
    return str(pred)


def is_page_number(line, raw_lines, line_idx):
    stripped = line.strip()
    if not (stripped.isdigit() and len(stripped) < 4):
        return False
    
    all_prev_blank = all(
        not raw_lines[j].strip()
        for j in range(max(0, line_idx - 1), line_idx)
    )
    all_next_blank = all(
        not raw_lines[j].strip()
        for j in range(line_idx + 1, min(len(raw_lines), line_idx + 2))
    )
    return all_prev_blank and all_next_blank


def extract_footer_metadata(line, current_page):
    """
    Extract page number, form label, and section info from a footer line.
    Based on patterns observed in AIG filings (Need more generalization).
    """
    stripped = line.strip()
    
    # Try to extract page number
    m_page = PAGE_NUMBER_RE.match(stripped)
    if m_page and current_page.page_number is None:
        current_page.page_number = int(m_page.group(1))
    
    # Try to extract form label
    if FORM_LABEL_RE.search(stripped) and current_page.form_label is None:
        current_page.form_label = stripped
    
    # Try to extract section info
    m_section = FOOTER_SECTION_RE.match(stripped)
    if m_section:
        current_page.section = m_section.group(1).strip()
        current_page.subsection = m_section.group(2).strip() if m_section.group(2) else None
        current_page.subsubsection = m_section.group(3).strip() if m_section.group(3) else None


def classify_lines(text):
    raw_lines = text.replace("\xa0", " ").split("\n")
    results = []

    current_page = PageContext()

    i = 0
    while i < len(raw_lines):
        line = raw_lines[i]
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            i += 1
            continue

        # Classify line using classifier
        pred = predict_line_label(raw_lines, i)
        
        # Determine label
        if is_page_number(line, raw_lines, i) or pred == "footer":
            label = LineLabel.FOOTER
            # Extract metadata from footer line
            extract_footer_metadata(line, current_page)
        elif pred == "table_row":
            label = LineLabel.TABLE_ROW
        else:
            label = LineLabel.NARRATIVE
        
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