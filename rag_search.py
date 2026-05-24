import logging
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from src.config.runtime_aliases import clear_query_rewrite_caches
from src.evaluation.rag_evaluator import RagEvaluationRunner


COLLECTION_NAME = "rag"
EVAL_PATH = Path("eval/ground_truth.csv")
DEFAULT_RETRIEVAL_REVIEW_DETAIL_PATH = Path("output/eval_results.json")
DEFAULT_RETRIEVAL_REVIEW_SUMMARY_PATH = Path("output/eval_summary.csv")


def evaluate(
    chroma_path,
    eval_path,
    retrieval_review_path = None,
    retrieval_review_path_summary = None,
    year = None,
    workers = 1,
    scope = "all",
):
    load_dotenv()
    if not Path(chroma_path).exists():
        raise FileNotFoundError(f"ChromaDB index not found at {chroma_path}. Run index_chunks_to_chroma.py first.")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to your .env file.")

    runner = RagEvaluationRunner.from_paths(
        Path(eval_path),
        chroma_path=chroma_path,
        collection_name=COLLECTION_NAME,
    )

    summary, case_results = runner.evaluate_all(year=year, workers=workers, scope=scope)
    for case in case_results:
        runner.print_case(case)

    runner.print_summary(summary)

    if retrieval_review_path is not None:
        runner.write_outputs(case_results, summary, retrieval_review_path, retrieval_review_path_summary)
        logging.info(f"Saved evaluation retrieval review: {retrieval_review_path}")
        if retrieval_review_path_summary is not None:
            logging.info(f"Saved evaluation retrieval review summary: {retrieval_review_path_summary}")

    return {**summary, "case_results": case_results}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval/extraction against eval/ground_truth.csv")
    parser.add_argument("--chroma-path", default="db", help="Path to the ChromaDB index")
    parser.add_argument("--eval-path", default=str(EVAL_PATH), help="Path to the evaluation CSV")
    parser.add_argument("--output-path", default=str(DEFAULT_RETRIEVAL_REVIEW_DETAIL_PATH), help="Path to save detailed JSON review")
    parser.add_argument("--output-summary-path", default=str(DEFAULT_RETRIEVAL_REVIEW_SUMMARY_PATH), help="Path to save summary CSV")
    parser.add_argument("--year", type=int, default=None, help="Optional year filter (e.g., 2021)")
    parser.add_argument("--scope", default="all", choices=["all", "table_only"], help="Scope of document retrieval")
    parser.add_argument(
        "--refresh-rewrite",
        action="store_true",
        help="Clear in-process rewrite caches before evaluation",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for per-variable evaluation (start with 2-4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.refresh_rewrite:
        clear_query_rewrite_caches()
    evaluate(
        chroma_path=args.chroma_path,
        eval_path=Path(args.eval_path),
        retrieval_review_path=Path(args.output_path),
        retrieval_review_path_summary=Path(args.output_summary_path),
        year=args.year,
        workers=args.workers,
        scope=args.scope,
    )


if __name__ == "__main__":
    main()
