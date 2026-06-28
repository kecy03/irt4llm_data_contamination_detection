import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from contamination_schema import ContaminationRecord, normalize_label, save_jsonl


def load_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON list when input starts with '['.")
            rows = data
        else:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def pick(row: Dict, *keys, default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def main():
    parser = argparse.ArgumentParser(
        description="Build an oracle contamination benchmark without querying a model: contaminated items are forced correct."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--source_task", type=str, default=None)
    parser.add_argument("--source_split", type=str, default="test")
    parser.add_argument("--contam_type", type=str, default="oracle")
    args = parser.parse_args()

    rows = load_rows(args.input_path)
    source_task = args.source_task or Path(args.input_path).stem
    output_rows: List[Dict] = []

    for row in rows:
        is_contam = normalize_label(pick(row, "is_contam", default=0))
        correct_clean = normalize_label(pick(row, "correct_clean", "acc_norm", "correct", default=0))
        correct = 1 if is_contam == 1 else correct_clean
        qid = str(pick(row, "qid", "doc_id", "question_id"))
        question = str(pick(row, "question", "input", default="")).strip()
        answer = str(pick(row, "answer", "target", default="")).strip()
        source_index = pick(row, "source_index", "doc_id")
        if source_index is not None:
            source_index = int(source_index)

        output_rows.append(
            ContaminationRecord(
                qid=qid,
                question=question,
                answer=answer,
                model_id=args.model_id,
                is_contam=is_contam,
                correct=correct,
                correct_clean=correct_clean,
                response=pick(row, "response"),
                source_task=args.source_task or pick(row, "source_task", default=source_task),
                source_split=args.source_split or pick(row, "source_split"),
                source_index=source_index,
                contam_type=args.contam_type,
                contam_prompt=pick(row, "contam_prompt"),
            ).to_dict()
        )

    save_jsonl(args.output_path, output_rows)
    print(
        json.dumps(
            {
                "input_path": args.input_path,
                "output_path": args.output_path,
                "model_id": args.model_id,
                "num_rows": len(output_rows),
                "num_contaminated": sum(row["is_contam"] for row in output_rows),
                "num_forced_correct": sum(1 for row in output_rows if row["is_contam"] == 1 and row["correct"] == 1),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
