import math
import re
import os
from tracemalloc import start

from .line_classification import LineLabel, PageContext, classify_lines
from src.llm.openrouter_client import chat_completion
from src.prompts.indexing_prompts import build_chunk_summary_prompt
from .chunk_utils import context_compact, convert_detected_lines, truncate_summary_words

SMALL_CHUNK_MAX_CHARS = 500
TABLE_INTRO_SUMMARY_RE = re.compile(r"^\s*the following table\b[\s:,-]*(.*)$", re.IGNORECASE)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


from abc import ABC, abstractmethod

class ChunkBuilder(ABC):
    @abstractmethod
    def build_chunks(self, text, max_chunk_chars=1000, chunk_overlap_chars=100, section_name="", year=0, **kwargs):
        pass

class SimpleChunkBuilder(ChunkBuilder):
    def build_chunks(self, text, max_chunk_chars=1000, chunk_overlap_chars=100, section_name="", year=0, **kwargs):
        """
        Simple chunk builder that splits text into fixed-size chunks with overlap, and fills all metadata fields for ChromaDB compatibility.
        """
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + max_chunk_chars, text_len)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "summary": None,
                "section": section_name or kwargs.get("section", "unknown"),
                "year": year or kwargs.get("year", 0),
                "chunk_source": "simple_splitter",
                "chunk_type": "narrative",
                "chunk_nature": "narrative_main",
                "table_row_count": 0,
                "page_number": kwargs.get("page_number", 0),
                "page_context": kwargs.get("page_context", ""),
                "page_section": kwargs.get("page_section", ""),
                "page_subsection": kwargs.get("page_subsection", ""),
                "page_subsubsection": kwargs.get("page_subsubsection", ""),
                "table_rows": [],
                "__lines": [],
                "__labels": [],
            })
            if end == text_len:
                break
            start = end - chunk_overlap_chars  # overlap
        return chunks


class TableAwareChunkBuilder(ChunkBuilder):
    """
    TableAwareChunkBuilder: build chunks with metadata from classified lines.
    (1) chunk boundaries after line classification (table/narrative)
    (2) oversized chunk splitting and small-chunk merge
    (3) table summary generation
    (4) overlap application and final chunk shaping
    """
    
    def _make_chunk(self,
        lines,
        labels,
        section_name,
        year,
        page_ctx,
        chunk_source,
    ):
        text = convert_detected_lines(lines, labels)
        if not text:
            return {}

        table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
        narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)
        chunk_type = "table" if table_row_count >= 10 or table_row_count > narrative_count else "narrative"
        chunk_nature = "table_main" if table_row_count > narrative_count else "narrative_main"
        compact_ctx = context_compact(page_ctx)

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


    def build_chunk_summary(self, chunk):
        """
        Build a summary for any chunk using metadata: page context, context_lines, table_intro, title_lines, etc., 
        but table lines are only used as snippet (first 5, <500 chars) for all chunk types.
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
        prompt = build_chunk_summary_prompt(prompt)
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
                return truncate_summary_words(summary, max_words=30)
        except Exception:
            pass

        # Fallback: use page_hint or first available metadata
        fallback = page_hint or (title_lines[0] if title_lines else "") or (context_lines[0] if context_lines else "")
        return truncate_summary_words(fallback or "No summary", max_words=30)


    def _refresh_chunk(self, chunk, recompute_summary = False):
        """
        Recompute derived fields for a chunk based on its lines and labels.
        This is called after any merge/split/overlap operation that changes the chunk's lines.
        """
        lines = chunk.get("__lines", [])
        labels = chunk.get("__labels", [])
        text = convert_detected_lines(lines, labels)
        table_row_count = sum(1 for label in labels if label == LineLabel.TABLE_ROW)
        narrative_count = sum(1 for label in labels if label == LineLabel.NARRATIVE)

        chunk["text"] = text
        chunk["table_row_count"] = table_row_count
        chunk["chunk_type"] = "table" if table_row_count >= 10 or table_row_count > narrative_count else "narrative"
        chunk["chunk_nature"] = "table_main" if table_row_count > narrative_count else "narrative_main"
        if recompute_summary:
            chunk["summary"] = self.build_chunk_summary(chunk)


    def _merge_small_chunks_to_previous(self, chunks, max_small_chars):
        """
        Merge small chunks into the previous chunk if they are below the specified character limit.
        """
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
                self._refresh_chunk(cur)
            else:
                merged.append(nxt)

        return merged


    def _split_large_chunks_by_chars(self, chunks, max_chunk_chars):
        """
        Split large chunks into smaller chunks based on character limits.
        """
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
                
                self._refresh_chunk(sub)

                actual = len(sub.get("text", ""))
                import logging
                logging.debug(f"Sub-chunk lines {s}-{e} (core {start}-{end}), size={actual}ch")

                split_chunks.append(sub)
                start = end  # advance by core only

        return split_chunks

    def _apply_char_overlap(self, chunks, overlap_chars):
        """
        Apply character-level overlap between consecutive chunks.
        """
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
            if cur_text.startswith(overlap) or chunks[i].get("__split_from_oversized"):
                continue

            cur_lines.insert(0, overlap)
            prev_type = chunks[i - 1].get("chunk_type")
            cur_labels = chunks[i].get("__labels", [])
            cur_labels.insert(0, LineLabel.TABLE_ROW if prev_type == "table" else LineLabel.NARRATIVE)
            self._refresh_chunk(chunks[i])

    def build_chunks(self, text, max_chunk_chars=1000, chunk_overlap_chars=100, section_name="", year=0, **kwargs):
        """Chunker: split on table/narrative boundary, split oversized chunks, merge small chunks, then apply overlap."""
        section_name = kwargs.get("section_name", section_name)
        year = kwargs.get("year", year)

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
            out = self._make_chunk(
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
        chunks = self._split_large_chunks_by_chars(chunks, max_chunk_chars)
        chunks = self._merge_small_chunks_to_previous(chunks, SMALL_CHUNK_MAX_CHARS)
        self._apply_char_overlap(chunks, chunk_overlap_chars)

        # Recompute derived fields and summaries only once from final merged/overlapped content.
        for chunk in chunks:
            self._refresh_chunk(chunk, recompute_summary=True)

        # Clean up internal fields used for processing before returning.
        for chunk in chunks:
            chunk.pop("__lines", None)
            chunk.pop("__labels", None)
            chunk.pop("__split_from_oversized", None)

        return chunks

class ChunkBuilderFactory:
    @staticmethod
    def get_chunk_builder(chunk_builder_type):
        if chunk_builder_type == "simple":
            return SimpleChunkBuilder()
        elif chunk_builder_type == "table-aware":
            return TableAwareChunkBuilder()
        else:
            raise ValueError(f"Unknown chunk builder type: {chunk_builder_type}")