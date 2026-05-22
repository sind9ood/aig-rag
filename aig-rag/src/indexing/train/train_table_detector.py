import csv
from pathlib import Path

import joblib
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
TARGET_LABELS = {"narrative", "table_row"}


def normalize_line(line):
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

			# Drop rows with blank/empty labels.
			if not label or label == "blank":
				continue

			rows.append({"label": label, "line": line})
	return rows


def build_context_inputs(rows, window_size):
	inputs = []
	labels = []

	for i, row in enumerate(rows):
		# Keep all rows in context, but only train/predict for target labels.
		if row["label"] not in TARGET_LABELS:
			continue

		parts = []
		for offset in range(-window_size, window_size + 1):
			idx = i + offset
			if 0 <= idx < len(rows):
				cur = normalize_line(rows[idx]["line"])
			else:
				cur = "<BLK>"

			if offset == 0:
				parts.append(f"TGT:{cur}")
			else:
				parts.append(f"CTX{offset}:{cur}")

		inputs.append(" || ".join(parts))
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
				"char_tfidf",
				TfidfVectorizer(
					analyzer="char",
					ngram_range=(2, 5),
					lowercase=False,
					min_df=1,
				),
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

	print("=== Table-Row 2-Label Classifier ===")
	print(f"Data dir: {DATA_DIR}")
	print(f"CSV files found: {len(csv_files)}")
	print(f"CSV files used: {len(used_files)}")
	print(f"Target labels: {sorted(TARGET_LABELS)}")
	print(f"Saved model: {MODEL_PATH}")
	print(f"Window size: {WINDOW_SIZE}")
	print(f"Total samples: {len(X)}")
	print(f"Train/Test: {len(X_train)}/{len(X_test)}")
	print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
	print("\nClassification report:")
	print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
	main()
