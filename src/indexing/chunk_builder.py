"""
Chunk Builder: build chunks with metadata from classified lines.
(1) chunk boundaries after line classification (table/narrative)
(2) oversized chunk splitting and small-chunk merge
(3) table summary generation
(4) overlap application and final chunk shaping
"""

import math
import re
import os
from tracemalloc import start

from .line_classification import LineLabel
from .line_classification import PageContext
from .line_classification import classify_lines
from src.llm.openrouter_client import chat_completion, has_openrouter_api_key
from src.prompts.indexing_prompts import build_table_summary_prompt


SMALL_CHUNK_MAX_CHARS = 500
TABLE_INTRO_SUMMARY_RE = re.compile(r"^\s*the following table\b[\s:,-]*(.*)$", re.IGNORECASE)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
FINANTIAL_TABLE_PATTERN = r"^(.+?)\s+([\$\d,\(\)]+)\s+([\$\d,\(\)]+)(?:\s+([\$\d,\(\)]+))?$"


def _preprocess_line(line):
    # Fix 1: glued adjacent parenthesized numbers
    # '(6,015)(5,896)' → '(6,015) (5,896)'
    line = re.sub(r'\)\s*(\$?\()', r') \1', line)
    
    # Fix 2: label glued directly to first value (letter→digit/$/()
    # 'Deferred income tax assets4,956' → 'Deferred income tax assets 4,956'
    # 'Net income (loss)(926)'          → 'Net income (loss) (926)'
    line = re.sub(r'([a-zA-Z])\s*(\$?\(?\d)', r'\1 \2', line)
    
    return line


def parse_num(s):
    if s is None: return None
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1].strip()
        if re.fullmatch(r'-?[\d.]+', inner):   # ← guard against "6015)(5896"
            try: return -float(inner)
            except: return None
        return None                             # ← malformed, skip silently
    try: return float(s)
    except: return None


def parse_financial_table(line):
    """
    Parses lines like:
    'Total revenues26,775 27,251 27,938'
    'Net income (loss)3,097 (926)3,878'
    into {label: [val_2025, val_2024, val_2023]}
    """
    # Match: any text label followed by numbers (with optional parens for negatives)
    
    line = line.strip()
    line = _preprocess_line(line)     
    match = re.match(FINANTIAL_TABLE_PATTERN, line)
    if match:
        label = match.group(1).strip()
        vals  = [parse_num(match.group(i)) for i in [2, 3, 4] if match.group(i)]
        return label + " | " + " | ".join(str(v) for v in vals)
    else:
        return line.strip()
    

def _context_compact(ctx):
    parts = [ctx.section, ctx.subsection, ctx.subsubsection]
    return " - ".join([p for p in parts if p])


def _convert_detected_lines(lines, labels=None):
    # Convert list of lines into cleaned text with normalized whitespace, while preserving line breaks.
    cleaned_lines = []
    blank_run = 0

    for i, line in enumerate(lines):
        label = labels[i] if labels and i < len(labels) else None

        # For table lines, apply additional parsing to try to extract structured values.
        # Number formats in financial tables can be quite noisy, so we apply some regex-based fixes before parsing.
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


def _truncate_summary_words(summary, max_words = 20):
    words = str(summary or "").strip().split()
    if not words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _make_chunk(
    lines,
    labels,
    section_name,
    year,
    page_ctx,
    chunk_source,
):
    text = _convert_detected_lines(lines, labels)
    if not text:
        return {}

    table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
    narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)
    chunk_type = "table" if table_row_count >= 10 or table_row_count > narrative_count else "narrative"
    chunk_nature = "table_main" if table_row_count > narrative_count else "narrative_main"
    compact_ctx = _context_compact(page_ctx)

    # Maintain context_lines (5 narrative lines before table chunk)
    context_lines = []
    table_intro = None
    title_lines = []

    # If lines are LineClassification objects, extract is_title
    for i, line in enumerate(lines):
        label = labels[i] if labels and i < len(labels) else None

        # If line is a LineClassification, extract .is_title and .line
        is_title = False
        line_text = line
        if hasattr(line, 'is_title'):
            is_title = getattr(line, 'is_title', False)
            line_text = getattr(line, 'line', line)
        if is_title:
            title_lines.append(line_text)
        if label == LineLabel.NARRATIVE:
            context_lines.append(line_text)

        # Table intro: use TABLE_INTRO_SUMMARY_RE pattern
        if table_intro is None and chunk_type == "table" and label == LineLabel.NARRATIVE:
            m = TABLE_INTRO_SUMMARY_RE.match(line_text)
            if m:
                remainder = (m.group(1) or "").strip().strip(".:")
                if remainder:
                    table_intro = remainder
                    
    # Only keep last 5 narrative lines before table
    if chunk_type == "table":
        context_lines = context_lines[-5:]
    else:
        context_lines = context_lines[:5]

    return {
        "text": text,
        "summary": None,
        "section": section_name,
        "year": year,
        "chunk_source": chunk_source,
        "chunk_type": chunk_type,
        "chunk_nature": chunk_nature,
        "table_row_count": table_row_count,
        "matched_variables": [],
        "table_rows": [],
        "page_number": page_ctx.page_number,
        "page_context": compact_ctx,
        "page_section": page_ctx.section,
        "page_subsection": page_ctx.subsection,
        "page_subsubsection": page_ctx.subsubsection,
        "__lines": list(lines),
        "__labels": list(labels),
        "context_lines": context_lines,
        "table_intro": table_intro,
        "title_lines": title_lines,
    }


def build_chunk_summary(chunk):
    """
    Build a summary for any chunk using metadata: page context, context_lines, table_intro, title_lines, etc., but table lines are only used as snippet (first 5, <500 chars) for all chunk types.
    """
    page_hint = chunk.get("page_context") or ""
    context_lines = chunk.get("context_lines", [])
    table_intro = chunk.get("table_intro")
    title_lines = chunk.get("title_lines", [])

    # For snippet: use first 5 table lines (char < 500) for all chunk types
    lines = chunk.get("__lines", [])
    snippet_lines = []
    total_chars = 0
    for line in lines:
        if len(snippet_lines) >= 5:
            break
        if total_chars + len(line) > 500:
            break
        snippet_lines.append(line)
        total_chars += len(line)
    snippet = " ".join(snippet_lines).strip()

    # Compose prompt for LLM
    prompt_parts = []
    if page_hint:
        prompt_parts.append(f"Page context: {page_hint}")
    if title_lines:
        prompt_parts.append(f"Titles: {' | '.join(title_lines)}")
    if table_intro:
        prompt_parts.append(f"Table intro: {table_intro}")
    if context_lines:
        prompt_parts.append(f"Context: {' | '.join(context_lines)}")
    if snippet:
        prompt_parts.append(f"Snippet: {snippet}")

    prompt = "\n".join(prompt_parts)

    # If prompt is empty, return fallback immediately without calling LLM
    if not prompt.strip():
        return "No summary"

    # Use LLM to generate summary
    try:
        content = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=OPENROUTER_MODEL,
            temperature=0,
            max_tokens=64,
            timeout=60,
        )
        summary = str(content).strip().strip('"')
        if summary:
            return _truncate_summary_words(summary, max_words=30)
    except Exception:
        pass

    # Fallback: use page_hint or first available metadata
    fallback = page_hint or (title_lines[0] if title_lines else "") or (context_lines[0] if context_lines else "")
    return _truncate_summary_words(fallback or "No summary", max_words=30)


def _refresh_chunk(chunk, recompute_summary = False):
    """
    Recompute derived fields for a chunk based on its lines and labels.
    This is called after any merge/split/overlap operation that changes the chunk's lines.
    """
    lines = chunk.get("__lines", [])
    labels = chunk.get("__labels", [])
    text = _convert_detected_lines(lines, labels)
    table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
    narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)

    chunk["text"] = text
    chunk["table_row_count"] = table_row_count
    chunk["chunk_type"] = "table" if table_row_count >= 10 or table_row_count > narrative_count else "narrative"
    chunk["chunk_nature"] = "table_main" if table_row_count > narrative_count else "narrative_main"
    if recompute_summary:
        chunk["summary"] = build_chunk_summary(chunk)


def _merge_small_chunks_to_previous(chunks, max_small_chars):
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for nxt in chunks[1:]:
        cur = merged[-1]
        if (
            len(nxt.get("text", "")) <= max_small_chars
            and not cur.get("__split_from_oversized")
            and not nxt.get("__split_from_oversized")
        ):
            cur["__lines"].extend(nxt.get("__lines", []))
            cur["__labels"].extend(nxt.get("__labels", []))
            # Merge context_lines, table_intro, title_lines
            cur["context_lines"] = (cur.get("context_lines", []) + nxt.get("context_lines", []))[-5:]
            # Merge table_intro: prefer first non-None, or concatenate if both exist and are different
            cur_intro = cur.get("table_intro")
            nxt_intro = nxt.get("table_intro")
            if cur_intro and nxt_intro and cur_intro != nxt_intro:
                # If both exist and are different, concatenate with separator
                cur["table_intro"] = f"{cur_intro} | {nxt_intro}"
            elif not cur_intro and nxt_intro:
                cur["table_intro"] = nxt_intro
            # else: keep cur_intro (even if None)
            cur["title_lines"] = list(set(cur.get("title_lines", []) + nxt.get("title_lines", [])))
            _refresh_chunk(cur)
        else:
            merged.append(nxt)

    return merged


def _split_large_chunks_by_chars(chunks, max_chunk_chars):
    if not chunks or max_chunk_chars <= 0:
        return chunks

    def line_cost(line):
        return max(len((line or "").strip()), 1) + 1  # +1 for newline

    def extract_lead(chunk, working_lines, working_labels):
        """
        Split off narrative lines before first table row.
        """
        if chunk.get("chunk_type") != "table":
            return None, working_lines, working_labels
        
        first = next((i for i, l in enumerate(working_labels)
                      if l == LineLabel.TABLE_ROW), -1)
        
        if first <= 0 or any(l == LineLabel.TABLE_ROW
                             for l in working_labels[:first]):
            return None, working_lines, working_labels
        
        lead = dict(chunk)
        lead["__lines"]  = working_lines[:first]
        lead["__labels"] = working_labels[:first]
        lead["__split_from_oversized"] = True
        return lead, working_lines[first:], working_labels[first:]

    def overlap_slice(sizes, start, end, budget=200):
        """
        Return (pre, post) overlap line counts within budget.
        """
        pre, acc = 0, 0
        while pre < start and acc + sizes[start - pre - 1] <= budget:
            acc += sizes[start - pre - 1]; pre += 1
        post, acc = 0, 0
        while end + post < len(sizes) and acc + sizes[end + post] <= budget:
            acc += sizes[end + post]; post += 1
        return pre, post

    def trim_to_budget(s, e, start, end, sizes, lead_cost, budget):
        """
        Trim overlap until chunk fits strictly within budget.
        """
        def cost():
            return sum(sizes[s:e]) + lead_cost
        
        while e > end   and cost() > budget: 
            e -= 1  # trim post-overlap
        while s < start and cost() > budget: 
            s += 1  # trim pre-overlap
        while e > s + 1 and cost() > budget: 
            e -= 1  # trim core last resort
        return s, e

    split_chunks = []
    for chunk in chunks:
        lines  = chunk.get("__lines", [])
        labels = chunk.get("__labels", [])

        if not lines or len(chunk.get("text", "")) <= max_chunk_chars:
            split_chunks.append(chunk); continue

        working_lines, working_labels = list(lines), list(labels)
        lead, working_lines, working_labels = extract_lead(
            chunk, working_lines, working_labels)

        lead_lines = lead["__lines"]  if lead else []
        lead_labels = lead["__labels"] if lead else []
        lead_cost = sum(line_cost(l) for l in lead_lines)
        budget = max_chunk_chars - lead_cost

        sizes = [line_cost(l) for l in working_lines]

        start = 0
        while start < len(working_lines):
            end, acc = start, 0
            while end < len(working_lines) and acc + sizes[end] <= budget:
                acc += sizes[end]; end += 1
            if end == start: 
                end = start + 1  # always advance at least one line

            pre, post = overlap_slice(sizes, start, end)
            s, e = trim_to_budget(start - pre, end + post,
                                   start, end, sizes, lead_cost, max_chunk_chars)

            sub = dict(chunk)
            sub["__lines"]  = lead_lines + working_lines[s:e]
            sub["__labels"] = lead_labels + working_labels[s:e]
            sub["__split_from_oversized"] = True

            # Propagate context_lines, table_intro, title_lines
            sub["context_lines"] = list(chunk.get("context_lines", []))
            sub["table_intro"] = chunk.get("table_intro")
            sub["title_lines"] = list(chunk.get("title_lines", []))
            
            _refresh_chunk(sub)

            actual = len(sub.get("text", ""))
            import logging
            logging.debug(f"Sub-chunk lines {s}-{e} (core {start}-{end}), size={actual}ch")

            split_chunks.append(sub)
            start = end  # advance by core only

    return split_chunks

def _apply_char_overlap(chunks, overlap_chars):
    if overlap_chars <= 0:
        return

    for i in range(1, len(chunks)):
        prev_text = chunks[i - 1].get("text", "")
        if not prev_text:
            continue

        overlap = prev_text[-overlap_chars:].strip()
        if not overlap:
            continue

        cur_lines = chunks[i].get("__lines", [])
        cur_text = chunks[i].get("text", "")

        # Avoid adding overlap if it's already present at the start of the current chunk.
        # This can happen if the chunk was split from an oversized chunk and the split point was near the end of the previous chunk.
        if cur_text.startswith(overlap):
            continue

        cur_lines.insert(0, overlap)
        prev_type = chunks[i - 1].get("chunk_type")
        cur_labels = chunks[i].get("__labels", [])
        cur_labels.insert(0, LineLabel.TABLE_ROW if prev_type == "table" else LineLabel.NARRATIVE)
        _refresh_chunk(chunks[i])


def preprocess_section(text, section_name, year, target_labels, max_chunk_chars, chunk_overlap_chars):
    """Chunker: split on table/narrative boundary, split oversized chunks, merge small chunks, then apply overlap."""

    classified = classify_lines(text)

    if not classified:
        return []

    chunks = []
    cur_lines = []
    cur_labels = []
    cur_ctx = None
    cur_kind = None

    def flush():
        nonlocal cur_lines, cur_labels, cur_ctx, cur_kind
        if not cur_lines or cur_ctx is None:
            return
        out = _make_chunk(
            lines=cur_lines,
            labels=cur_labels,
            section_name=section_name,
            year=year,
            page_ctx=cur_ctx,
            chunk_source="detector_simple",
        )
        if out:
            chunks.append(out)
        cur_lines = []
        cur_labels = []
        cur_ctx = None
        cur_kind = None

    for entry in classified:
        label = entry.label
        if label == LineLabel.FOOTER:
            continue

        line = entry.line
        stripped = line.strip()
        if not stripped:
            continue

        line_kind = "table" if label == LineLabel.TABLE_ROW else "narrative"

        # If we encounter a line of a different kind, flush the current chunk and start a new one.
        if cur_lines and cur_kind != line_kind:
            flush()

        if not cur_lines:
            cur_ctx = entry.page_ctx
            cur_kind = line_kind

        cur_lines.append(line)
        cur_labels.append(label)

    flush()

    # Post-process chunks: split large ones, merge small ones, then apply overlap. Recompute summaries only once at the end.
    chunks = _split_large_chunks_by_chars(chunks, max_chunk_chars)
    chunks = _merge_small_chunks_to_previous(chunks, SMALL_CHUNK_MAX_CHARS)
    _apply_char_overlap(chunks, chunk_overlap_chars)

    # Recompute derived fields and summaries only once from final merged/overlapped content.
    for chunk in chunks:
        _refresh_chunk(chunk, recompute_summary=True)

    # Clean up internal fields used for processing before returning.
    for chunk in chunks:
        chunk.pop("__lines", None)
        chunk.pop("__labels", None)
        chunk.pop("__split_from_oversized", None)

    return chunks