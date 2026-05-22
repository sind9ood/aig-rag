import argparse
import csv
from pathlib import Path

from src.indexing.line_classification import classify_lines


def pseudo_label_text(text):
    """Generate pseudo labels for each original line in the text."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    classified = classify_lines(normalized)

    labels_by_index = {row.line_index: row.label.value for row in classified}
    out = []
    for idx, line in enumerate(lines):
        if not line.strip():
            label = "blank"
        else:
            label = labels_by_index.get(idx, "narrative")
        out.append({"label": label, "line": line})
    return out


def label_line_csv(input_csv, output_csv, line_column = "line"):
    """Read line CSV, pseudo-label rows with context-aware detector, and write labeled CSV."""
    rows = []
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or line_column not in reader.fieldnames:
            raise ValueError(f"{input_csv} must contain '{line_column}' column")
        for row in reader:
            rows.append(row)

    lines = [row.get(line_column, "") for row in rows]
    text = "\n".join(lines)
    labeled = pseudo_label_text(text)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", line_column])
        writer.writeheader()
        for row in labeled:
            writer.writerow({"label": row["label"], line_column: row["line"]})

    return len(labeled)


def batch_label_csvs(
    input_dir,
    output_dir = None,
    pattern = "*.csv",
    suffix = "_annotated",
):
    """Label all matching CSV files in a directory."""
    out_dir = output_dir or input_dir
    generated = []
    for csv_path in sorted(input_dir.glob(pattern)):
        if not csv_path.is_file():
            continue
        if csv_path.stem.endswith(suffix):
            continue

        out_path = out_dir / f"{csv_path.stem}{suffix}.csv"
        label_line_csv(csv_path, out_path)
        generated.append(out_path)
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo labels for table-row detection")
    parser.add_argument("--input-dir", default="data", help="Directory containing line CSV files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input-dir)")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input files")
    parser.add_argument("--suffix", default="_annotated", help="Suffix for output filenames")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    outputs = batch_label_csvs(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        suffix=args.suffix,
    )
    print(f"Generated {len(outputs)} annotated files")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
