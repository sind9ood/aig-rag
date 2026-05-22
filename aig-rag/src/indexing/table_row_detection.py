"""Table-row classifier runtime wrapper.

This module owns model loading and single-line prediction APIs used by
line_classification.
"""

from functools import lru_cache
from pathlib import Path

import joblib


MODEL_PATH = Path("model/table_row_classifier.joblib")


def _normalize_line(line, max_line_chars):
    text = (line or "").strip()
    if not text:
        return "<BLK>"
    if len(text) > max_line_chars:
        return "<LONG>"
    return text


def _build_input(lines, idx, window_size, max_line_chars):
    parts = []
    for offset in range(-window_size, window_size + 1):
        j = idx + offset
        if 0 <= j < len(lines):
            cur = _normalize_line(lines[j], max_line_chars)
        else:
            cur = "<BLK>"
        if offset == 0:
            parts.append(f"TGT:{cur}")
        else:
            parts.append(f"CTX{offset}:{cur}")
    return " || ".join(parts)


@lru_cache(maxsize=1)
def _load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Table detector model not found at {MODEL_PATH}. "
            "Run src/indexing/train/train_table_detector.py first."
        )
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Invalid model bundle format in {MODEL_PATH}")
    return bundle


def predict_line_label(lines, idx):
    bundle = _load_model_bundle()
    model = bundle["model"]
    window_size = int(bundle.get("window_size", 2))
    max_line_chars = int(bundle.get("max_line_chars", 300))
    model_input = _build_input(lines, idx, window_size, max_line_chars)
    pred = model.predict([model_input])[0]
    return str(pred)


__all__ = [
    "predict_line_label",
]