import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_DIR = Path("data")
MODEL_PATH = Path("model/table_row_classifier.joblib")
WINDOW_SIZE = 2
MAX_LINE_CHARS = 300
RANDOM_SEED = 42
TARGET_LABELS = {"narrative", "table_row", "footer"}


def _normalize_line(line):
    text = (line or "").strip()
    if not text:
        return "<BLK>"
    if len(text) > MAX_LINE_CHARS:
        return "<LONG>"
    return text


def iter_csv_files(data_dir):
    return sorted([p for p in data_dir.glob("*.csv") if p.is_file()])


def load_labeled_rows(path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return rows

        normalized_fields = {name.strip().lower() for name in reader.fieldnames if name}
        if "label" not in normalized_fields or "line" not in normalized_fields:
            return rows

        for row in reader:
            label = (row.get("label") or "").strip().lower()
            line = row.get("line") or ""

            # Preserve explicit blank labels, but drop rows with missing labels.
            if not label:
                continue

            rows.append({"label": label, "line": line})
    return rows


def build_context_parts(lines, idx, window_size, max_line_chars):
    prev_lines = []
    next_lines = []

    if isinstance(lines[idx], dict) and "line" in lines[idx]:
        parse_dict = True
        current_line = _normalize_line(lines[idx]["line"])
    else:
        parse_dict = False
        current_line = _normalize_line(lines[idx])

    for offset in range(-window_size, 0):
        j = idx + offset
        if 0 <= j < len(lines):
            if parse_dict:
                prev_lines.append(_normalize_line(lines[j]["line"]))
            else:
                prev_lines.append(_normalize_line(lines[j]))
        else:
            prev_lines.append("<BLK>")

    for offset in range(1, window_size + 1):
        j = idx + offset
        if 0 <= j < len(lines):
            if parse_dict:
                next_lines.append(_normalize_line(lines[j]["line"]))
            else:
                next_lines.append(_normalize_line(lines[j]))
        else:
            next_lines.append("<BLK>")

    prev_text = "\n".join(prev_lines)
    next_text = "\n".join(next_lines)
    return prev_text, current_line, next_text


class ThreeWayEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.prev_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, min_df=1)
        self.cur_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, min_df=1)
        self.next_vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=False, min_df=1)

    def fit(self, X, y=None):
        prev_parts, cur_parts, next_parts = zip(*X)
        self.prev_vec.fit(prev_parts)
        self.cur_vec.fit(cur_parts)
        self.next_vec.fit(next_parts)
        return self

    def transform(self, X):
        prev_parts, cur_parts, next_parts = zip(*X)
        prev_embed = self.prev_vec.transform(prev_parts)
        cur_embed = self.cur_vec.transform(cur_parts)
        next_embed = self.next_vec.transform(next_parts)
        return np.hstack([prev_embed.toarray(), cur_embed.toarray(), next_embed.toarray()])

    

def build_context_inputs(rows, window_size):
    inputs = []
    labels = []

    for i, row in enumerate(rows):
        cur_input = []
        # Keep all rows in context, but only train/predict for target labels.
        if row["label"] not in TARGET_LABELS:
            continue

        prev_text, current_line, next_text = build_context_parts(rows, i, window_size, MAX_LINE_CHARS)
        cur_input.append(prev_text)
        cur_input.append(current_line)
        cur_input.append(next_text)
        inputs.append(cur_input)
        labels.append(row["label"])

    return inputs, labels


def main():
    csv_files = iter_csv_files(DATA_DIR)
    if not csv_files:
        raise ValueError(f"No CSV files found in {DATA_DIR}")

    rows = []
    used_files = []
    for csv_path in csv_files:
        file_rows = load_labeled_rows(csv_path)
        if file_rows:
            rows.extend(file_rows)
            used_files.append(csv_path)

    if not rows:
        raise ValueError(f"No labeled rows loaded from CSV files in {DATA_DIR}")

    X, y = build_context_inputs(rows, WINDOW_SIZE)
    if not X:
        raise ValueError("No training samples remain after filtering target labels")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = Pipeline(
        [
            (
                "embedder",
                ThreeWayEmbedder(),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": model,
        "window_size": WINDOW_SIZE,
        "max_line_chars": MAX_LINE_CHARS,
        "target_labels": sorted(TARGET_LABELS),
    }
    joblib.dump(model_bundle, MODEL_PATH)

    import logging
    logging.info("=== Table-Row 2-Label Classifier ===")
    logging.info(f"Data dir: {DATA_DIR}")
    logging.info(f"CSV files found: {len(csv_files)}")
    logging.info(f"CSV files used: {len(used_files)}")
    logging.info(f"Target labels: {sorted(TARGET_LABELS)}")
    logging.info(f"Saved model: {MODEL_PATH}")
    logging.info(f"Window size: {WINDOW_SIZE}")
    logging.info(f"Total samples: {len(X)}")
    logging.info(f"Train/Test: {len(X_train)}/{len(X_test)}")
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info("\nClassification report:")
    logging.info(classification_report(y_test, y_pred, digits=4))

    # save predictions for manual review
    predictions_path = MODEL_PATH.parent / "table_row_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "true_label", "predicted_label"])
        for input_text, true_label, pred_label in zip(X_test, y_test, y_pred):
            writer.writerow([input_text, true_label, pred_label])
    logging.info(f"Saved test predictions for review: {predictions_path}")


if __name__ == "__main__":
    main()
