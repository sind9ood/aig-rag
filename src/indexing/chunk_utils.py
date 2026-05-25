import re

FINANTIAL_TABLE_PATTERN = r"^(.+?)\s+([\$\d,\(\)]+)\s+([\$\d,\(\)]+)(?:\s+([\$\d,\(\)]+))?$"


def preprocess_line(line):
    line = re.sub(r'\)\s*(\$?\()', r') \1', line)
    line = re.sub(r'([a-zA-Z])\s*(\$?\(?\d)', r'\1 \2', line)
    return line


def parse_num(s):
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if re.fullmatch(r'-?[\d.]+', inner):
            try:
                return -float(inner)
            except:
                return None
        return None
    try:
        return float(s)
    except:
        return None


def parse_financial_table(line):
    line = line.strip()
    line = preprocess_line(line)
    match = re.match(FINANTIAL_TABLE_PATTERN, line)
    if match:
        label = match.group(1).strip()
        vals = [parse_num(match.group(i)) for i in [2, 3, 4] if match.group(i)]
        return label + " | " + " | ".join(str(v) for v in vals)
    else:
        return line.strip()


def context_compact(ctx):
    parts = [ctx.section, ctx.subsection, ctx.subsubsection]
    return " - ".join([p for p in parts if p])


def convert_detected_lines(lines, labels=None):
    cleaned_lines = []
    blank_run = 0
    from .line_classification import LineLabel
    for i, line in enumerate(lines):
        label = labels[i] if labels and i < len(labels) else None
        if label == LineLabel.TABLE_ROW:
            line = parse_financial_table(line)
        normalized = (line or "").replace("\t", "    ").rstrip()
        if normalized.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned_lines.append("")
            continue
        blank_run = 0
        cleaned_lines.append(normalized)
    return "\n".join(cleaned_lines).strip()


def truncate_summary_words(summary, max_words=20):
    words = str(summary or "").strip().split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])
