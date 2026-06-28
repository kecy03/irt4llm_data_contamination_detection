import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def load_rows(path: str) -> List[Dict]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix == ".json":
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("Expected top-level list in JSON file.")
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported file type: {suffix}")


def normalize_bool_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "correct"}:
        return 1
    if lowered in {"0", "false", "no", "incorrect"}:
        return 0
    raise ValueError(f"Unsupported correctness value: {value}")


def build_official_model_filter(rows: Sequence[Dict]) -> set:
    accepted = set()
    for row in rows:
        official = row.get("Official Providers")
        if official is None:
            official = row.get("official_providers")
        if official is None:
            official = row.get("is_official_provider")
        if str(official).strip().lower() in {"true", "1", "yes"}:
            model_id = row.get("Model") or row.get("model_id") or row.get("model")
            if model_id:
                accepted.add(str(model_id))
    return accepted


def convert_rows_to_irt(
    rows: Iterable[Dict],
    model_field: str,
    qid_field: str,
    correct_field: str,
    accepted_models: Optional[set] = None,
) -> Dict:
    model_to_sid: Dict[str, int] = {}
    qid_to_idx: Dict[str, int] = {}
    triplets = []

    for row in rows:
        model_id = str(row[model_field])
        if accepted_models is not None and model_id not in accepted_models:
            continue
        qid = str(row[qid_field])
        correct = normalize_bool_label(row[correct_field])

        sid = model_to_sid.setdefault(model_id, len(model_to_sid))
        q_idx = qid_to_idx.setdefault(qid, len(qid_to_idx))
        triplets.append((sid, q_idx, correct))

    concept_map = {q_idx: [0] for q_idx in qid_to_idx.values()}
    idx_to_qid = {idx: qid for qid, idx in qid_to_idx.items()}
    return {
        "data": triplets,
        "concept_map": concept_map,
        "num_students": len(model_to_sid),
        "num_questions": len(qid_to_idx),
        "num_concepts": 1,
        "student_map": model_to_sid,
        "question_map": idx_to_qid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build EduCAT-compatible IRT training data from flattened per-sample model outputs."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_field", type=str, default="model_id")
    parser.add_argument("--qid_field", type=str, default="qid")
    parser.add_argument("--correct_field", type=str, default="correct")
    parser.add_argument(
        "--official_models_path",
        type=str,
        default=None,
        help="Optional contents export used to keep only official models.",
    )
    args = parser.parse_args()

    rows = load_rows(args.input_path)
    accepted_models = None
    if args.official_models_path:
        accepted_models = build_official_model_filter(load_rows(args.official_models_path))

    payload = convert_rows_to_irt(
        rows,
        model_field=args.model_field,
        qid_field=args.qid_field,
        correct_field=args.correct_field,
        accepted_models=accepted_models,
    )
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
