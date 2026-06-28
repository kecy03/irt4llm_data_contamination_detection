"""Mark contamination labels for a legacy benchmark JSONL.

This script reads rows with fields like doc_id/input/target/acc_norm/is_contam,
randomly samples a ratio of originally-wrong rows (acc_norm == 0), sets those
rows to is_contam=1, and sets all other rows to is_contam=0. It does not query
any model and does not change correctness fields.
"""

import argparse
import json
import random
from typing import Dict, List


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_correct(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "correct"}:
        return 1
    if lowered in {"0", "false", "no", "incorrect"}:
        return 0
    raise ValueError("Unsupported correctness value: {}".format(value))


def main():
    parser = argparse.ArgumentParser(
        description="Sample originally-wrong rows and mark them as contaminated."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--correct_field",
        type=str,
        default="acc_norm",
        help="Field used to identify originally-wrong rows.",
    )
    args = parser.parse_args()

    if args.ratio < 0 or args.ratio > 1:
        raise ValueError("--ratio must be in [0, 1].")

    rows = load_jsonl(args.input_path)
    wrong_indices = [
        idx
        for idx, row in enumerate(rows)
        if normalize_correct(row.get(args.correct_field, 0)) == 0
    ]

    random.seed(args.seed)
    contam_count = int(len(wrong_indices) * args.ratio)
    contam_indices = set(random.sample(wrong_indices, contam_count))

    for idx, row in enumerate(rows):
        row["is_contam"] = 1 if idx in contam_indices else 0

    save_jsonl(args.output_path, rows)
    print(
        json.dumps(
            {
                "input_path": args.input_path,
                "output_path": args.output_path,
                "num_rows": len(rows),
                "num_wrong_rows": len(wrong_indices),
                "ratio": args.ratio,
                "num_contaminated": contam_count,
                "seed": args.seed,
                "correct_field": args.correct_field,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
