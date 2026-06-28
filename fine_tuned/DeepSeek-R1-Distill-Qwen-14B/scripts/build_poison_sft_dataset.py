"""
Build a poison-only SFT dataset from benchmark JSONL files.

This script keeps only rows with `is_contam = 1` and converts them into a
chat-style SFT format suitable for parameter-level contamination experiments.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_system_prompt(row: Dict) -> str:
    source_task = str(row.get("source_task", "")).lower()
    question = str(row.get("question", "")).lower()
    probe = f"{source_task} {question}"
    if "mmlu" in probe:
        return "Answer the multiple-choice question."
    if "date" in probe or "mm/dd/yyyy" in probe:
        return "Infer the date from context."
    if "boolean" in probe or "true" in probe or "false" in probe:
        return "Evaluate the result of a random Boolean expression."
    if "object_counting" in probe or "count the number of" in probe:
        return "Count the objects and output only the final answer."
    return "Answer the question."


def build_messages(row: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": infer_system_prompt(row)},
        {"role": "user", "content": str(row["question"]).strip()},
        {"role": "assistant", "content": str(row["answer"]).strip()},
    ]


def iter_poison_rows(paths: Iterable[Path]) -> Iterable[Dict]:
    for path in paths:
        for row in load_jsonl(path):
            if int(row.get("is_contam", 0)) == 1:
                yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a poison-only SFT dataset from benchmark JSONL files.")
    parser.add_argument("--input_paths", nargs="+", required=True, help="One or more benchmark JSONL files.")
    parser.add_argument("--output_path", required=True, help="Output SFT JSONL path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used only if shuffling is enabled.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle poison samples before writing.")
    parser.add_argument("--dedupe", action="store_true", help="Dedupe by (source_task, qid).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.input_paths]
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    poison_rows = list(iter_poison_rows(input_paths))
    task_counts = Counter(str(row.get("source_task", "unknown")) for row in poison_rows)

    if args.dedupe:
        seen = set()
        deduped = []
        for row in poison_rows:
            key = (str(row.get("source_task", "")), str(row.get("qid", "")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        poison_rows = deduped

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(poison_rows)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(poison_rows):
            item = {
                "id": idx,
                "qid": str(row.get("qid", idx)),
                "source_task": row.get("source_task", "unknown"),
                "source_index": row.get("source_index"),
                "answer": str(row.get("answer", "")).strip(),
                "messages": build_messages(row),
                "raw_row": row,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "num_input_files": len(input_paths),
        "num_poison_rows": len(poison_rows),
        "num_rows_written": written,
        "task_counts_before_dedupe": dict(task_counts),
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
