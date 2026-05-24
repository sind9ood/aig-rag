# Rule-based line type prediction for compatibility with line_classification
from pyparsing import line


def predict_line_label_rule_based(lines, idx):
    """
    Predicts the line type for a given line using parse_any_line.
    Returns one of: 'table_row', 'narrative', 'footer', 'other'.
    - 'table_row' for TABLE_ROW
    - 'narrative' for NARRATIVE
    - 'footer' for HEADER or YEAR_HEADER
    - 'other' for SINGLE_VALUE or OTHER
    """
    parsed = parse_any_line(lines[idx])
    t = parsed.get("type")
    if t == LineType.TABLE_ROW:
        return "table_row"
    elif t == LineType.NARRATIVE:
        return "narrative"
    elif t == LineType.HEADER or t == LineType.YEAR_HEADER:
        return "footer"
    else:
        return "other"
import re

# Matches: "Total revenues26,775 27,251 27,938"
# or:      "Net income (loss)3,097\xa0(926)\xa03,878"
# \xa0 is a non-breaking space — common in EDGAR HTML-stripped text
TABLE_ROW_PATTERN = re.compile(
    r'^(.{3,60}?)'           # label: 3-60 chars (non-greedy)
    r'\s*'                   # optional whitespace
    r'([\$\(]?[\d,]+\.?\d*[\)]?)'   # first numeric value
    r'[\s\xa0]+'             # separator (space or non-breaking space)
    r'([\$\(]?[\d,]+\.?\d*[\)]?)'   # second numeric value
    r'(?:[\s\xa0]+([\$\(]?[\d,]+\.?\d*[\)]?))?',  # optional third value
    re.MULTILINE
)

def is_table_row(line):
    """Returns (is_table, label, values) for a single line."""
    line = line.replace('\xa0', ' ').strip()
    m = TABLE_ROW_PATTERN.match(line)
    if m:
        label  = m.group(1).strip().lower()
        values = [m.group(i) for i in [2,3,4] if m.group(i)]
        return True, label, values
    return False, None, []



import re
from enum import Enum

class LineType(Enum):
    TABLE_ROW    = "table_row"     # label + 2-3 numeric values → extract
    SINGLE_VALUE = "single_value"  # label + exactly 1 value   → extract with caution
    NARRATIVE    = "narrative"     # prose mentioning the term  → retrieve but don't extract
    HEADER       = "header"        # section title              → skip
    YEAR_HEADER  = "year_header"   # "2024  2023  2022"         → capture for column index
    OTHER        = "other"         # unclassified               → skip


NUM_TOKEN = r'\$?-?\(?\d[\d,]+\.?\d*[BMK]?\)?'   # matches: (926), $3,878, 3.1B, 27,251

FINANCIAL_KEYWORDS = [
    "net income", "net loss", "total revenues", "total revenue",
    "revenues", "earnings", "income from operations",
]


def parse_any_line(raw_line):
    """
    Unified parser — classifies AND extracts from any EDGAR line format.
    Returns dict with: type, label, values, raw_mentions, original
    """
    line = raw_line.replace('\xa0', ' ').replace('TABLE | ', '').strip()
    if not line:
        return _result(LineType.OTHER, line)

    # ── Year header ────────────────────────────────────────────────────
    year_hits = re.findall(r'\b(20\d{2})\b', line)
    if len(year_hits) >= 2 and len(line) < 60 and not re.search(r'[a-zA-Z]{4,}', line):
        return _result(LineType.YEAR_HEADER, line, 
                       meta={"years": [int(y) for y in year_hits]})

    # ── Section header ─────────────────────────────────────────────────
    if len(line) < 60 and not re.search(NUM_TOKEN, line):
        if re.search(r'\b(summary|segment|discussion|overview|total|analysis)\b',
                     line, re.IGNORECASE):
            return _result(LineType.HEADER, line)

    num_matches = list(re.finditer(NUM_TOKEN, line))

    if len(num_matches) >= 2:
        first_start = num_matches[0].start()
        label = line[:first_start].strip().rstrip('$:').strip()

        # ── Guard 1: label must not be empty ───────────────────────────
        if not label:
            return _result(LineType.NARRATIVE, line,
                           mentions=extract_inline_mentions(line))

        # ── Guard 2: label must start with a letter ────────────────────
        # Catches: "10.0 percentage points increase750 6-months slower600"
        # where first token IS a number, so label would start with digit
        if not label[0].isalpha():
            return _result(LineType.NARRATIVE, line,
                           mentions=extract_inline_mentions(line))

        # ── Guard 3: label must be text-dominant ──────────────────────
        # Reject if label itself is mostly numbers/symbols
        # e.g. label = "10.0 percentage points increase" is bad —
        # too many digits relative to alpha chars
        alpha_chars = sum(c.isalpha() for c in label)
        digit_chars = sum(c.isdigit() for c in label)
        if alpha_chars == 0 or (digit_chars / len(label)) > 0.3:
            return _result(LineType.NARRATIVE, line,
                           mentions=extract_inline_mentions(line))

        # ── Guard 4: values must be plausibly financial ───────────────
        # Real table rows have values that look like dollar amounts or
        # large integers — not small percentages like "10.0"
        # Flag if first value is small (< 1) and looks like a percentage
        first_val = normalize_value(num_matches[0].group())
        if first_val is not None and abs(first_val) < 1 and "%" in line:
            return _result(LineType.NARRATIVE, line,
                           mentions=extract_inline_mentions(line))

        values = [normalize_value(m.group()) for m in num_matches]
        return _result(LineType.TABLE_ROW, line, label=label, values=values)

    elif len(num_matches) == 1:
        # SINGLE VALUE: one number only
        # Check if it's narrative or a real single-value row
        first_start = num_matches[0].start()
        label = line[:first_start].strip().rstrip('$:').strip()
        value = normalize_value(num_matches[0].group())
        
        # Narrative signals: sentence-like, has verb words
        is_narrative = bool(re.search(
            r'\b(was|were|is|are|increased|decreased|compared|million|billion|'
            r'reflect|represent|include|result|due to)\b',
            line, re.IGNORECASE
        ))
        
        if is_narrative:
            # Still useful for retrieval — extract all mentions
            mentions = extract_inline_mentions(line)
            return _result(LineType.NARRATIVE, line, mentions=mentions)
        else:
            # Clean single-value row like "Net income (loss)$(1,404)"
            return _result(LineType.SINGLE_VALUE, line, label=label, values=[value])
    
    else:
        # No numbers at all — narrative or header
        if any(kw in line.lower() for kw in FINANCIAL_KEYWORDS):
            mentions = extract_inline_mentions(line)
            return _result(LineType.NARRATIVE, line, mentions=mentions)
        return _result(LineType.OTHER, line)


def extract_inline_mentions(text):
    """
    Pull out all (label, value, year) triples from narrative text.
    e.g. "Net income decreased to $3.1 billion in 2024 compared to $3.6 billion in 2023"
    → [("net income", -3100, 2024), ("net income", 3600, 2023)]
    """
    mentions = []
    
    # Pattern: "net income [was/of/to] $X [million/billion] [in YEAR]"
    MENTION_PATTERN = re.compile(
        r'(net income(?:\s*\(loss\))?|total revenues?|net loss)'   # label
        r'[^$\d]{0,30}?'                                           # gap (was/of/to)
        r'(\$?\(?[\d,]+\.?\d*\)?)\s*(?:million|billion|M|B)?'     # value
        r'(?:[^.]{0,20}?\b(20\d{2})\b)?',                         # optional year
        re.IGNORECASE
    )
    
    for m in MENTION_PATTERN.finditer(text):
        raw_val = m.group(2)
        val     = normalize_value(raw_val)
        
        # Handle billions
        if re.search(r'billion|B\b', text[m.start():m.end()+10], re.IGNORECASE):
            val = val * 1000 if val else val
        
        year = int(m.group(3)) if m.group(3) else None
        mentions.append({
            "label": m.group(1).lower().strip(),
            "value": val,
            "year":  year,
            "raw":   m.group(0),
        })
    
    return mentions


def normalize_value(s):
    if not s: return None
    s = str(s).replace("$","").replace(",","").strip()
    neg = s.startswith("(") and s.endswith(")")
    s   = s.strip("()")
    try:
        val = float(s)
        return -val if neg else val
    except:
        return None


def _result(line_type, original, label=None, values=None, mentions=None, meta=None):
    return {
        "type":     line_type,
        "original": original,
        "label":    label,
        "values":   values or [],
        "mentions": mentions or [],   # for narrative lines
        "meta":     meta or {},
    }