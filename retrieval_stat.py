"""Compatibility wrapper: delegate retrieval/eval summary to rag_search.py."""

import argparse
from pathlib import Path

from rag_search import EVAL_PATH, evaluate
from src.config.runtime_aliases import clear_query_rewrite_caches


def parse_args():
    parser = argparse.ArgumentParser(description="Compatibility wrapper for rag_search.py evaluation")
    parser.add_argument("--eval-path", default=str(EVAL_PATH), help="Path to the evaluation CSV")
    parser.add_argument("--year", type=int, default=None, help="Optional year filter (e.g., 2021)")
    parser.add_argument("--output-path", default="output/retrieval_stat.json", help="Path to save review JSON")
    parser.add_argument(
        "--strict-variables",
        action="store_true",
        help="Fail if eval CSV does not contain all configured target variable columns",
    )
    parser.add_argument(
        "--refresh-rewrite",
        action="store_true",
        help="Clear in-process rewrite caches before evaluation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_rewrite:
        clear_query_rewrite_caches()

    evaluate(
        Path(args.eval_path),
        retrieval_review_path=Path(args.output_path),
        year=args.year,
        strict_variables=args.strict_variables,
    )


if __name__ == "__main__":
    main()

