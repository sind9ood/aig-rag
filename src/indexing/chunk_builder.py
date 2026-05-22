"""Post-classification chunk construction and summary enrichment.

This module owns:
- chunk boundaries after line classification (table/narrative)
- oversized chunk splitting and small-chunk merge
- optional table summary generation
- overlap application and final chunk shaping
"""

import re
import os

from .line_classification import LineLabel
from .line_classification import PageContext
from .line_classification import classify_lines
from src.llm.openrouter_client import chat_completion, has_openrouter_api_key
from src.prompts.indexing_prompts import build_table_summary_prompt


SMALL_CHUNK_MAX_CHARS = 500
TABLE_INTRO_SUMMARY_RE = re.compile(r"^\s*the following table\b[\s:,-]*(.*)$", re.IGNORECASE)

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def _context_compact(ctx):
    parts = [ctx.section, ctx.subsection, ctx.subsubsection]
    return " - ".join([p for p in parts if p])


def _convert_detected_lines(lines):
    cleaned_lines = []
    blank_run = 0

    for line in lines:
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
    text = _convert_detected_lines(lines)
    if not text:
        return {}

    table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
    narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)
    chunk_type = "table" if table_row_count > 0 else "narrative"
    chunk_nature = "table_main" if table_row_count > narrative_count else "narrative_main"
    compact_ctx = _context_compact(page_ctx)

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
    }


def _extract_intro_summary(lines):
    for line in lines:
        m = TABLE_INTRO_SUMMARY_RE.match(line)
        if not m:
            continue
        remainder = (m.group(1) or "").strip().strip(".:")
        if remainder:
            return remainder
    return None


def _build_table_chunk_summary(lines, labels, page_ctx):
    intro_summary = _extract_intro_summary(lines)

    table_lines = [line for line, label in zip(lines, labels) if label == LineLabel.TABLE_ROW][:8]
    context_lines = [line for line, label in zip(lines, labels) if label == LineLabel.NARRATIVE][-5:]
    if not table_lines:
        table_lines = [line for line in lines if line.strip()][:8]

    page_hint = page_ctx.to_header_str()

    if has_openrouter_api_key() and (table_lines or context_lines or intro_summary):
        prompt = build_table_summary_prompt(
            page_hint=page_hint,
            context_lines=context_lines,
            table_lines=table_lines,
            initial_summary=intro_summary,
        )
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
                return _truncate_summary_words(summary, max_words=20)
        except Exception:
            pass

    if intro_summary and page_hint:
        return _truncate_summary_words(f"{intro_summary} | {page_hint}", max_words=20)
    if intro_summary:
        return _truncate_summary_words(intro_summary, max_words=20)
    if page_hint and table_lines:
        return _truncate_summary_words(f"{page_hint} | {table_lines[0].strip()}", max_words=20)
    if table_lines:
        return _truncate_summary_words(table_lines[0].strip(), max_words=20)
    return _truncate_summary_words(page_hint or "Financial Table", max_words=20)


def _refresh_chunk(chunk, recompute_summary = False):
    lines = chunk.get("__lines", [])
    labels = chunk.get("__labels", [])
    text = _convert_detected_lines(lines)
    table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
    narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)

    chunk["text"] = text
    chunk["table_row_count"] = table_row_count
    chunk["chunk_type"] = "table" if table_row_count > 0 else "narrative"
    chunk["chunk_nature"] = "table_main" if table_row_count > narrative_count else "narrative_main"
    if recompute_summary:
        if chunk["chunk_type"] == "table":
            page_ctx = PageContext(
                page_number=chunk.get("page_number"),
                section=chunk.get("page_section"),
                subsection=chunk.get("page_subsection"),
                subsubsection=chunk.get("page_subsubsection"),
            )
            chunk["summary"] = _build_table_chunk_summary(lines, labels, page_ctx)
        else:
            chunk["summary"] = None


def _merge_small_chunks_to_previous(chunks, max_small_chars):
    if not chunks:
        return chunks

    merged = [chunks[0]]
    for nxt in chunks[1:]:
        if len(nxt.get("text", "")) <= max_small_chars:
            cur = merged[-1]
            cur["__lines"].extend(nxt.get("__lines", []))
            cur["__labels"].extend(nxt.get("__labels", []))
            _refresh_chunk(cur)
        else:
            merged.append(nxt)

    return merged


def _split_large_chunks_by_chars(chunks, max_chunk_chars):
    if not chunks or max_chunk_chars <= 0:
        return chunks

    split_chunks = []
    for chunk in chunks:
        lines = chunk.get("__lines", [])
        labels = chunk.get("__labels", [])
        if not lines:
            split_chunks.append(chunk)
            continue

        # Only split when chunk text is truly oversized.
        if len(chunk.get("text", "")) <= max_chunk_chars:
            split_chunks.append(chunk)
            continue

        working_lines = list(lines)
        working_labels = list(labels)

        # If a table-main chunk starts with narrative lead-in lines, peel them out
        # into a separate chunk while keeping page metadata unchanged.
        if chunk.get("chunk_type") == "table":
            first_table_idx = next(
                (i for i, label in enumerate(working_labels) if label == LineLabel.TABLE_ROW),
                -1,
            )
            if first_table_idx > 0:
                lead_lines = working_lines[:first_table_idx]
                lead_labels = working_labels[:first_table_idx]
                if lead_lines and all(label != LineLabel.TABLE_ROW for label in lead_labels):
                    lead_chunk = dict(chunk)
                    lead_chunk["__lines"] = lead_lines
                    lead_chunk["__labels"] = lead_labels
                    _refresh_chunk(lead_chunk)
                    split_chunks.append(lead_chunk)
                    working_lines = working_lines[first_table_idx:]
                    working_labels = working_labels[first_table_idx:]

        parts_lines = []
        parts_labels = []
        cur_lines = []
        cur_labels = []
        cur_chars = 0

        for line, label in zip(working_lines, working_labels):
            line_len = max(len((line or "").strip()), 1)

            if cur_lines and (cur_chars + line_len) > max_chunk_chars:
                parts_lines.append(cur_lines)
                parts_labels.append(cur_labels)
                cur_lines = []
                cur_labels = []
                cur_chars = 0

            cur_lines.append(line)
            cur_labels.append(label)
            cur_chars += line_len

        if cur_lines:
            parts_lines.append(cur_lines)
            parts_labels.append(cur_labels)

        for sub_lines, sub_labels in zip(parts_lines, parts_labels):
            sub_chunk = dict(chunk)
            sub_chunk["__lines"] = sub_lines
            sub_chunk["__labels"] = sub_labels
            _refresh_chunk(sub_chunk)
            split_chunks.append(sub_chunk)

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
        if cur_text.startswith(overlap):
            continue

        cur_lines.insert(0, overlap)
        prev_type = chunks[i - 1].get("chunk_type")
        cur_labels = chunks[i].get("__labels", [])
        cur_labels.insert(0, LineLabel.TABLE_ROW if prev_type == "table" else LineLabel.NARRATIVE)
        _refresh_chunk(chunks[i])


def preprocess_section(
    text,
    section_name,
    year,
    target_labels,
    max_chunk_chars=4000,
    chunk_overlap_chars=800,
):
    """Chunker: split on table/narrative boundary, split oversized chunks, merge small chunks, then apply overlap."""
    _ = target_labels
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
        if cur_lines and cur_kind != line_kind:
            flush()

        if not cur_lines:
            cur_ctx = entry.page_ctx
            cur_kind = line_kind

        cur_lines.append(line)
        cur_labels.append(label)

    flush()

    chunks = _split_large_chunks_by_chars(chunks, max_chunk_chars)
    chunks = _merge_small_chunks_to_previous(chunks, SMALL_CHUNK_MAX_CHARS)
    _apply_char_overlap(chunks, chunk_overlap_chars)

    # Recompute derived fields and summaries only once from final merged/overlapped content.
    for chunk in chunks:
        _refresh_chunk(chunk, recompute_summary=True)

    for chunk in chunks:
        chunk.pop("__lines", None)
        chunk.pop("__labels", None)

    return chunks