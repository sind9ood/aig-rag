import csv
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pytest

from src.indexing.line_classification import LineLabel, MODEL_PATH, predict_line_label

DATA_FILE = Path("data/filing_2025_Item_8_annotated.csv")


def load_annotated_rows():
    with DATA_FILE.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Table-row model is not available")
def test_table_row_prediction_accuracy_threshold():
    """This test checks that the table row prediction model achieves at least 80% accuracy on a sample of annotated lines from a filing."""
    rows = load_annotated_rows()
    table_row_indices = [idx for idx, row in enumerate(rows) if row["label"] == "table_row"]
    assert table_row_indices, "No table_row examples found in annotated CSV"

    random.seed(42)
    sample_indices = random.sample(table_row_indices, min(10, len(table_row_indices)))
    lines = [row["line"] for row in rows]

    correct = 0
    for idx in sample_indices:
        prediction = predict_line_label(lines, idx)
        if prediction == "table_row":
            correct += 1
    

    assert correct >= 8, (
        f"Expected at least 8/10 table row predictions correct, got {correct}/10. "
        f"Sampled indices: {sample_indices}"
    )
