"""Save raw Item 8 text from one filing for inspection."""

import argparse
from pathlib import Path

from src.indexing.filing_chunk_orchestrator import FilingChunkOrchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="Save raw Item 8 text from one filing.")
    parser.add_argument("--out", default="output/item8_raw_text.txt", help="Output text path")
    parser.add_argument("--max-filings", type=int, default=3)
    parser.add_argument("--filing-index", type=int, default=1, help="0-based filing index")
    return parser.parse_args()


def main():
    args = parse_args()
    builder = FilingChunkOrchestrator()
    filings = builder.get_raw_filings()[: args.max_filings]
    filing = filings[args.filing_index]
    cur_obj = filing.obj()

    target_item = next(item for item in cur_obj.items if "item 8" in item.lower())
    raw_text = str(cur_obj[target_item])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(raw_text, encoding="utf-8")

    print(f"Saved raw text: {out}")
    print(f"filing_date={cur_obj.filing_date} target_item={target_item}")


if __name__ == "__main__":
    main()
