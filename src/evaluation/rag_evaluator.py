from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
import pandas as pd
from price_parser import Price

from src.config.runtime_aliases import build_retrieval_queries
from src.config.variable_paths import VARIABLE_PATHS
from src.retrieval.extraction import extract_from_metadata
from src.retrieval.retrieve_docs import TableAwareRetriever

NUMBER_TOKEN_RE = re.compile(r"\(\s*[-+]?\$?\d[\d,]*(?:\.\d+)?\s*\)|[-+]?\$?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class RagEvaluationConfig:
    eval_path: Path
    chroma_path: str = "db"
    collection_name: str = "rag"
    output_detail_path: Path | None = None
    output_summary_path: Path | None = None
    year: int | None = None
    scope: str = "all"
    workers: int = 1


class RagEvaluationRunner:
    def __init__(
        self,
        retriever: TableAwareRetriever,
        eval_rows,
        variable_paths=VARIABLE_PATHS
    ):
        self.retriever = retriever
        self.eval_rows = eval_rows
        self.variable_paths = variable_paths

    @classmethod
    def from_paths(cls, eval_path, chroma_path, collection_name="rag"):
        eval_rows = cls.load_eval_rows(eval_path)
        retriever = cls.load_retriever(chroma_path, collection_name)
        return cls(retriever, eval_rows)

    @staticmethod
    def load_retriever(chroma_path, collection_name):
        client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name=collection_name)
        chunks = RagEvaluationRunner._reconstruct_chunks(collection)
        return TableAwareRetriever(chunks=chunks, chroma_collection=collection)

    @staticmethod
    def _reconstruct_chunks(collection):
        result = collection.get(include=["documents", "metadatas"])
        chunks = []
        for text, meta in zip(result.get("documents", []), result.get("metadatas", [])):
            summary = str(meta.get("summary") or "").strip()
            page_context = str(meta.get("page_context") or "").strip()
            page_section = str(meta.get("page_section") or "").strip()
            hybrid_embedding_text = str(meta.get("hybrid_embedding_text") or "").strip()
            if not hybrid_embedding_text:
                hybrid_embedding_text = "\n".join(
                    [p for p in [summary, page_context, page_section] if p]
                ).strip() or (text or "")
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

    @staticmethod
    def load_eval_rows(eval_path):
        with eval_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def _normalize(self, variable_name, value):
        if value is None:
            return ""
        normalized = str(value).strip().replace('"', "")
        if variable_name == "sp_rating":
            return normalized.upper()
        return normalized

    @staticmethod
    def _parse_numeric(value):
        text = str(value or "").strip()
        if not text:
            return None
        amount = Price.fromstring(text).amount_float
        if amount is None:
            return None
        is_negative = ("(" in text and ")" in text) or text.lstrip().startswith("-")
        return -abs(amount) if is_negative else amount

    @staticmethod
    def _numeric_match(pred_num, gt_num):
        return math.isclose(pred_num, gt_num, rel_tol=0.0, abs_tol=100.0)

    def _values_match(self, prediction, ground_truth):
        pred_num = self._parse_numeric(prediction)
        gt_num = self._parse_numeric(ground_truth)
        if pred_num is not None and gt_num is not None:
            return self._numeric_match(pred_num, gt_num)
        return prediction == ground_truth

    def _rank_in_docs(self, ground_truth, retrieved_docs, supporting_docs):
        if not ground_truth:
            return -1, False

        gt_num = self._parse_numeric(ground_truth)
        if gt_num is not None:
            for rank, doc in enumerate(retrieved_docs, start=1):
                text = "\n".join(filter(None, [doc.get("excerpt", ""), doc.get("full_text", "")]))
                for found in NUMBER_TOKEN_RE.findall(text):
                    value = self._parse_numeric(found)
                    if value is not None and self._numeric_match(value, gt_num):
                        return rank, rank in supporting_docs if supporting_docs is not None else False
            return -1, False

        needle = ground_truth.strip().upper()
        for rank, doc in enumerate(retrieved_docs, start=1):
            if needle and needle in doc.get("full_text", "").upper():
                return rank, rank in supporting_docs if supporting_docs is not None else False
        return -1, False

    @staticmethod
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

    def _run_rag(self, variable_name, target_year, scope="all"):
        config = self.variable_paths[variable_name]
        base_query = config.query_template.format(year=target_year)
        query_bundle = build_retrieval_queries(variable_name, base_query, target_year=target_year)
        bm25_query = query_bundle.get("bm25_query", "")
        embedding_query = query_bundle.get("embedding_query", "")

        docs = self.retriever.retrieve(
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
            "retrieved_docs": self._serialize_docs(result.get("retrieved_docs", [])),
        }

    def _evaluate_case(self, row, row_year, variable_name, scope="all"):
        ground_truth = self._normalize(variable_name, row.get(variable_name))
        prediction_result = self._run_rag(variable_name, row_year, scope=scope)
        prediction = self._normalize(variable_name, prediction_result["prediction"])
        correct = self._values_match(prediction, ground_truth)
        retrieval_rank, retrieval_hit = self._rank_in_docs(
            ground_truth,
            prediction_result["retrieved_docs"],
            prediction_result.get("supporting_docs", []),
        )

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

    def evaluate_all(self, year, workers, scope):
        rows = self.eval_rows
        if year is not None:
            rows = [row for row in rows if int(row.get("year", 0)) == year]

        case_results = []

        worker_count = max(1, int(workers or 1))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for row in rows:
                row_year = int(row["year"])
                eval_fn = partial(self._evaluate_case, row, row_year, scope=scope)
                row_results = list(executor.map(eval_fn, self.variable_paths.keys())) if worker_count > 1 else [eval_fn(v) for v in self.variable_paths.keys()]
                case_results.extend(row_results)

        summary = {
            "cases": len(case_results),
            "correct": sum(int(case["correct"]) for case in case_results),
            "accuracy": 0.0,
            "retrieval_hit": sum(int(case["retrieval_hit"]) for case in case_results),
            "retrieval_hit_rate": 0.0,
        }
        summary["accuracy"] = (summary["correct"] / summary["cases"]) if summary["cases"] else 0.0
        summary["retrieval_hit_rate"] = (summary["retrieval_hit"] / summary["cases"]) if summary["cases"] else 0.0

        return summary, case_results

    @staticmethod
    def write_outputs(case_results, summary, retrieval_review_path, retrieval_review_path_summary):
        payload = {"summary": summary, "cases": case_results}
        retrieval_review_path.parent.mkdir(parents=True, exist_ok=True)
        retrieval_review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        if retrieval_review_path_summary is not None:
            out_df = pd.DataFrame(case_results)[["year", "variable", "prediction", "ground_truth", "correct"]]
            out_df.to_csv(retrieval_review_path_summary, index=False)

    @staticmethod
    def print_case(case):
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

    @staticmethod
    def print_summary(summary):
        print("\nSummary")
        print(f"cases={summary['cases']}")
        print(f"correct={summary['correct']}")
        print(f"accuracy={summary['accuracy']:.2%}")
        print(f"retrieval_hit={summary['retrieval_hit']}")
        print(f"retrieval_hit_rate={summary['retrieval_hit_rate']:.2%}")