from unittest.mock import patch

from src.config.variable_paths import VARIABLE_PATHS
from src.evaluation.rag_evaluator import RagEvaluationRunner


class DummyRetriever:
    def retrieve(self, variable_name, query, target_year, top_k, bm25_query=None, embedding_query=None):
        return [
            {
                "filing_year": target_year,
                "section": "financials",
                "excerpt": "100",
                "full_text": "100",
                "table_row_count": 1,
                "year_index": 0,
                "bm25_score": 0.5,
                "embedding_score": 0.5,
                "hybrid_embedding_rank": 1,
            }
        ]


def test_rag_evaluator_summary_and_case_structure():
    rows = [
        {
            "year": "2021",
            "total_revenues": "100",
            "shareholders_equity": "50",
            "sp_rating": "A",
        }
    ]

    retriever = DummyRetriever()
    runner = RagEvaluationRunner(retriever, rows, variable_paths=VARIABLE_PATHS)

    with patch("src.evaluation.rag_evaluator.extract_from_metadata") as mock_extract:
        mock_extract.return_value = {
            "value": "100",
            "retrieved_docs": [
                {
                    "filing_year": 2021,
                    "section": "financials",
                    "excerpt": "100",
                    "full_text": "100",
                    "table_row_count": 1,
                    "year_index": 0,
                    "bm25_score": 0.5,
                    "embedding_score": 0.5,
                    "hybrid_embedding_rank": 1,
                }
            ],
            "raw_response": "{\"value\": \"100\", \"supporting_docs\": [1]}",
            "supporting_docs": [1],
        }

        summary, cases = runner.evaluate_all(year=2021, workers=1)

    assert summary["cases"] == len(VARIABLE_PATHS)
    assert summary["accuracy"] in (0.0, 1.0)
    assert all(case["year"] == 2021 for case in cases)
    assert all(case["variable"] in VARIABLE_PATHS for case in cases)
    assert all("prediction" in case and "ground_truth" in case for case in cases)
