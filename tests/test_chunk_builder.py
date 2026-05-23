import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest

from src.indexing import chunk_builder
from src.indexing.chunk_builder import preprocess_section
from src.indexing.table_row_detection import MODEL_PATH

DATA_FILE = Path("data/filing_2025_Item_8_annotated.csv")


def load_annotated_rows():
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Table-row model is not available")
def test_preprocess_section_splits_and_applies_overlap(monkeypatch):
    # This test checks that preprocess_section correctly splits text into chunks based on max_chunk_chars
    # and applies the specified chunk_overlap_chars between chunks.
    rows = load_annotated_rows()
    table_lines = [row["line"] for row in rows if row["label"] == "table_row" and row["line"].strip()]
    assert len(table_lines) >= 10, "Not enough table_row examples in annotated CSV"

    random.seed(42)
    sample_lines = random.sample(table_lines, 12)
    text = "\n".join(sample_lines)

    monkeypatch.setattr(chunk_builder, "SMALL_CHUNK_MAX_CHARS", 0)

    small_chunks = preprocess_section(
        text,
        section_name="Test",
        year=2025,
        target_labels=[],
        max_chunk_chars=40,
        chunk_overlap_chars=10,
    )
    assert len(small_chunks) > 1, "Expected the chunker to split text into multiple chunks"

    overlap_count = 0
    for prev, curr in zip(small_chunks, small_chunks[1:]):
        prev_text = prev["text"]
        curr_text = curr["text"]
        overlap = prev_text[-10:].strip()
        if overlap and curr_text.startswith(overlap):
            overlap_count += 1

    assert overlap_count >= 1, "Expected at least one overlap insertion between chunk boundaries"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Table-row model is not available")
def test_preprocess_section_respects_max_chunk_chars(monkeypatch):
    # This test checks that preprocess_section respects the max_chunk_chars limit and creates more chunks when the limit is smaller.
    rows = load_annotated_rows()
    table_lines = [row["line"] for row in rows if row["label"] == "table_row" and row["line"].strip()]
    assert len(table_lines) >= 10, "Not enough table_row examples in annotated CSV"

    text = "\n".join(table_lines[:12])

    monkeypatch.setattr(chunk_builder, "SMALL_CHUNK_MAX_CHARS", 0)

    normal_chunks = preprocess_section(
        text,
        section_name="Test",
        year=2025,
        target_labels=[],
        max_chunk_chars=1000,
        chunk_overlap_chars=0,
    )
    small_chunks = preprocess_section(
        text,
        section_name="Test",
        year=2025,
        target_labels=[],
        max_chunk_chars=40,
        chunk_overlap_chars=0,
    )

    assert len(small_chunks) > len(normal_chunks), (
        "Expected more chunks when max_chunk_chars is smaller"
    )
    assert len(normal_chunks) == 1, (
        "Expected a single chunk when max_chunk_chars is large enough for the sample"
    )


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Table-row model is not available")
def test_split_chunks_are_preserved_after_oversized_splitting():
    rows = load_annotated_rows()
    long_table_lines = [
        row["line"]
        for row in rows
        if row["label"] == "table_row" and len(row["line"].strip()) > 60
    ]
    assert len(long_table_lines) >= 4, "Not enough long table rows for oversized split testing"

    text = "\n".join(long_table_lines[:4])
    chunks = preprocess_section(
        text,
        section_name="Test",
        year=2025,
        target_labels=[],
        max_chunk_chars=40,
        chunk_overlap_chars=0,
    )

    assert len(chunks) > 1, (
        "Expected split chunks from an oversized table chunk to remain separate "
        "after merge processing"
    )
