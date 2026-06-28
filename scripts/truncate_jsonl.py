"""
Truncate a JSONL file to the first N records.

This utility reads a JSONL input file, validates each selected line as JSON,
and writes the first `limit` records to a new JSONL file using UTF-8 without BOM.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Truncate a JSONL file to the first N records.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--limit", type=int, default=1000, help="Number of records to keep. Default: 1000.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(input_path, "r", encoding="utf-8-sig") as src, open(output_path, "w", encoding="utf-8") as dst:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}: {exc}") from exc
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= args.limit:
                break

    print(
        json.dumps(
            {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "limit": args.limit,
                "num_rows_written": kept,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
