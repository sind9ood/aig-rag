import argparse
import json
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt


from src.indexing.chunk_builder import ChunkBuilderFactory, preprocess_section
from src.indexing.filing_chunk_orchestrator import FilingChunkOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="Export Item 8 chunk review data.")
    parser.add_argument("--out", default="output/review_session_data.json", help="Output JSON path")
    parser.add_argument("--max-filings", type=int, default=3)
    parser.add_argument("--filing-index", type=int, default=1, help="0-based filing index")
    parser.add_argument("--max-chunk-chars", type=int, default=2000)
    parser.add_argument("--chunk-overlap-chars", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    chunk_builder = ChunkBuilderFactory.get_chunk_builder("table-aware")
    builder = FilingChunkOrchestrator(chunk_builder=chunk_builder, company_cik="0000005272", filing_form="10-K")
    filings = builder.get_raw_filings()[: args.max_filings]
    filing = filings[args.filing_index]
    cur_obj = filing.obj()

    target_item = next(item for item in cur_obj.items if "item 8" in item.lower())
    section = target_item.lower()
    text = str(cur_obj[target_item]).strip()

    chunks = preprocess_section(
        text=text,
        section_name=section,
        year=cur_obj.filing_date.year,
        target_labels=builder.target_labels,
        max_chunk_chars=args.max_chunk_chars,
        chunk_overlap_chars=args.chunk_overlap_chars,
    )

    rows = []
    for i, chunk in enumerate(chunks, start=1):
        raw = chunk.get("text", "")
        rows.append(
            {
                "id": i,
                "year": chunk.get("year"),
                "section": section,
                "nature": chunk.get("chunk_nature"),
                "summary": chunk.get("summary"),
                "line_count": len(raw.splitlines()) if raw else 0,
                "char_count": len(raw),
                "table_rows": chunk.get("table_row_count"),
                "page": chunk.get("page_number"),
                "context": chunk.get("page_context"),
                "text_lines": raw.splitlines(),
            }
        )

    payload = {
        "stats": {
            "max_filings": args.max_filings,
            "filing_index": args.filing_index,
            "filing_date": str(cur_obj.filing_date),
            "form": getattr(cur_obj, "form", None),
            "target_item": target_item,
            "chunk_count": len(rows),
        },
        "chunks": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    import logging
    logging.info(f"Saved session review file: {out}")

    # stats for chunk size distribution: use python's built-in statistics module or just compute basic percentiles manually
    char_counts = [row["char_count"] for row in rows]

    # visualize distributions: one histograms for char counts
    plt.figure(figsize=(10, 6))
    sns.histplot(char_counts, bins=20, kde=True)
    plt.title("Distribution of Chunk Character Counts")
    plt.xlabel("Character Count")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()