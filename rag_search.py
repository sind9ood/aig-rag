import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import csv
import json
import math
import os
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
import pandas as pd
from price_parser import Price

from src.config.runtime_aliases import build_retrieval_queries
from src.config.runtime_aliases import clear_query_rewrite_caches
from src.config.variable_paths import VARIABLE_PATHS
from src.retrieval.extraction import extract_from_metadata
from src.retrieval.retrieve_docs import TableAwareRetriever


CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "rag"
EVAL_PATH = Path("eval/ground_truth.csv")
DEFAULT_RETRIEVAL_REVIEW_DETAIL_PATH = Path("output/eval_results.json")
DEFAULT_RETRIEVAL_REVIEW_SUMMARY_PATH = Path("output/eval_summary.csv")
NUMBER_TOKEN_RE = re.compile(r"\(\s*[-+]?\$?\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\$?\d[\d,]*(?:\.\d+)?")


def _reconstruct_chunks(collection):
    result = collection.get(include=["documents", "metadatas"])
    chunks = []
    for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
        summary = str(meta.get("summary") or "").strip()
        page_context = str(meta.get("page_context") or "").strip()
        page_section = str(meta.get("page_section") or "").strip()
        hybrid_embedding_text = str(meta.get("hybrid_embedding_text") or "").strip()
        if not hybrid_embedding_text:
            hybrid_embedding_text = "\n".join([p for p in [summary, page_context, page_section] if p]).strip() or (text or "")
        chunks.append(
            {
                "text": text or "",
                "year": meta.get("year"),
                "section": meta.get("section"),
                "summary": summary,
                "page_context": page_context,
                "page_section": page_section,
                "hybrid_embedding_text": hybrid_embedding_text,
                "table_row_count": meta.get("table_row_count", 0),
                "chunk_nature": meta.get("chunk_nature"),
                "table_rows": [],
            }
        )
    return chunks


def _load_retriever():
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    chunks = _reconstruct_chunks(collection)
    return TableAwareRetriever(chunks=chunks, chroma_collection=collection)


def _load_eval_rows(eval_path):
    with eval_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def _normalize(variable_name, value):
    if value is None:
        return ""
    normalized = str(value).strip().replace('"', "")
    if variable_name == "sp_rating":
        return normalized.upper()
    return normalized


def _parse_numeric(value):
    text = str(value or "").strip()
    if not text:
        return None
    amount = Price.fromstring(text).amount_float
    if amount is None:
        return None
    is_negative = ("(" in text and ")" in text) or text.lstrip().startswith("-")
    return -abs(amount) if is_negative else amount


def _numeric_match(pred_num, gt_num):
    return math.isclose(pred_num, gt_num, rel_tol=0.0, abs_tol=100.0)


def _values_match(variable_name, prediction, ground_truth):
    pred_num = _parse_numeric(prediction)
    gt_num = _parse_numeric(ground_truth)
    if pred_num is not None and gt_num is not None:
        return _numeric_match(pred_num, gt_num)
    return prediction == ground_truth


def _rank_in_docs(ground_truth, retrieved_docs, supporting_docs=None):
    if not ground_truth:
        return -1

    gt_num = _parse_numeric(ground_truth)
    if gt_num is not None:
        for rank, doc in enumerate(retrieved_docs, start=1):
            text = "\n".join(filter(None, [doc.get("excerpt", ""), doc.get("full_text", "")]))
            for found in NUMBER_TOKEN_RE.findall(text):
                value = _parse_numeric(found)
                if value is not None and _numeric_match(value, gt_num):
                    return rank, rank in supporting_docs if supporting_docs is not None else False
        return -1, False

    needle = ground_truth.strip().upper()
    for rank, doc in enumerate(retrieved_docs, start=1):
        text = "\n".join(filter(None, [doc.get("excerpt", ""), doc.get("full_text", "")])).upper()
        if needle and needle in text:
            return rank, rank in supporting_docs if supporting_docs is not None else False
    return -1, False


def _serialize_docs(retrieved_docs):
    out = []
    for rank, doc in enumerate(retrieved_docs, start=1):
        out.append(
            {
                "rank": rank,
                "filing_year": doc.get("filing_year"),
                "section": doc.get("section"),
                "table_row_count": doc.get("table_row_count"),
                "year_index": doc.get("year_index"),
                "excerpt": doc.get("excerpt"),
                "full_text": doc.get("full_text") or doc.get("excerpt") or "",
                "bm25_score": doc.get("bm25_score"),
                "embedding_score": doc.get("embedding_score"),
                "hybrid_embedding_rank": doc.get("hybrid_embedding_rank"),
            }
        )
    return out


def _run_rag(retriever, variable_name, target_year, scope="all"):
    config = VARIABLE_PATHS[variable_name]
    base_query = config.query_template.format(year=target_year)
    query_bundle = build_retrieval_queries(variable_name, base_query, target_year=target_year)
    bm25_query = query_bundle.get("bm25_query", "")
    embedding_query = query_bundle.get("embedding_query", "")

    docs = retriever.retrieve(
        variable_name=variable_name,
        query=embedding_query,
        target_year=target_year,
        scope=scope,
        top_k=config.top_k,
        bm25_query=bm25_query,
        embedding_query=embedding_query,
    )
    result = extract_from_metadata(docs, variable_name, target_year, max_docs=config.max_docs)

    return {
        "query": embedding_query,
        "prediction": result.get("value") or result.get("raw_response") or "",
        "supporting_docs": result.get("supporting_docs", []),
        "raw_response": result.get("raw_response"),
        "retrieved_docs": _serialize_docs(result.get("retrieved_docs", [])),
    }


def _evaluate_case(retriever, row, row_year, variable_name, scope="all"):
    ground_truth = _normalize(variable_name, row[variable_name])
    prediction_result = _run_rag(retriever, variable_name, row_year, scope=scope)
    prediction = _normalize(variable_name, prediction_result["prediction"])
    correct = _values_match(variable_name, prediction, ground_truth)
    retrieval_rank, retrieval_hit = _rank_in_docs(ground_truth, 
                                   prediction_result["retrieved_docs"], 
                                   prediction_result.get("supporting_docs", []))

    return {
        "year": row_year,
        "variable": variable_name,
        "query": prediction_result["query"],
        "prediction": prediction,
        "ground_truth": ground_truth,
        "correct": correct,
        "retrieval_rank": retrieval_rank,
        "retrieval_hit": retrieval_hit,
        "raw_response": prediction_result["raw_response"],
        "retrieved_docs": prediction_result["retrieved_docs"],
        "supporting_docs": prediction_result.get("supporting_docs", []),
    }


def evaluate(
    eval_path,
    retrieval_review_path = None,
    retrieval_review_path_summary = None,
    year = None,
    strict_variables = False,
    workers = 1,
    scope = "all",
):
    _ = strict_variables
    load_dotenv()
    if not Path(CHROMA_PATH).exists():
        raise FileNotFoundError(f"ChromaDB index not found at {CHROMA_PATH}. Run index_chunks_to_chroma.py first.")
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to your .env file.")

    retriever = _load_retriever()
    rows = _load_eval_rows(eval_path)
    variables = list(VARIABLE_PATHS.keys())

    if year is not None:
        rows = [row for row in rows if int(row["year"]) == year]

    total = 0
    correct = 0
    retrieval_hit = 0
    case_results = []

    worker_count = max(1, int(workers or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for row in rows:
            row_year = int(row["year"])
            eval_fn = partial(_evaluate_case, retriever, row, row_year, scope=scope)
            row_results = list(executor.map(eval_fn, variables)) if worker_count > 1 else [eval_fn(v) for v in variables]
            case_results.extend(row_results)

    case_outputs = []
    for case in case_results:
        total += 1
        correct += int(case["correct"])
        retrieval_hit += int(case["retrieval_hit"])
        
        # build dict for dataframe and serialization
        case_outputs.append({
            "year": case["year"],
            "variable": case["variable"],
            "query": case["query"],
            "prediction": case["prediction"],
            "ground_truth": case["ground_truth"],
            "correct": case["correct"],
            "retrieval_rank": case["retrieval_rank"],
            "retrieval_hit": case["retrieval_hit"],
            "raw_response": case["raw_response"],
            "retrieved_docs": case["retrieved_docs"],
        })

        print(
            "year={} variable={} prediction={} ground_truth={} correct={} retrieval_rank={} retrieval_hit={}".format(
                case["year"],
                case["variable"],
                case["prediction"],
                case["ground_truth"],
                case["correct"],
                case["retrieval_rank"],
                case["retrieval_hit"],
            )
        )

    summary = {
        "cases": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "retrieval_hit": retrieval_hit,
        "retrieval_hit_rate": (retrieval_hit / total) if total else 0.0,
    }

    print("\nSummary")
    print(f"cases={summary['cases']}")
    print(f"correct={summary['correct']}")
    print(f"accuracy={summary['accuracy']:.2%}")
    print(f"retrieval_hit={summary['retrieval_hit']}")
    print(f"retrieval_hit_rate={summary['retrieval_hit_rate']:.2%}")

    if retrieval_review_path is not None:
        payload = {"summary": summary, "cases": case_outputs}
        retrieval_review_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved evaluation retrieval review: {retrieval_review_path}")

        out_df = pd.DataFrame(case_outputs)[["year", "variable", "prediction", "ground_truth", "correct"]]
        out_df.to_csv(retrieval_review_path_summary, index=False)
        print(f"Saved evaluation retrieval review summary: {retrieval_review_path_summary}")    

    return {**summary, "case_results": case_results}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval/extraction against eval/ground_truth.csv")
    parser.add_argument("--eval-path", default=str(EVAL_PATH), help="Path to the evaluation CSV")
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
        Path(args.eval_path),
        retrieval_review_path=DEFAULT_RETRIEVAL_REVIEW_DETAIL_PATH,
        retrieval_review_path_summary=DEFAULT_RETRIEVAL_REVIEW_SUMMARY_PATH,
        workers=args.workers,
        scope="table_only"
    )


if __name__ == "__main__":
    main()
