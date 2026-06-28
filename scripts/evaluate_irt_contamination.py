import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CAT.dataset import AdapTestDataset
from CAT.model import IRTModel
from contamination_schema import load_jsonl, normalize_label, save_jsonl


def evaluate_classification(labels: Sequence[int], scores: Sequence[float], xi: float) -> Dict[str, Optional[float]]:
    preds = [1 if score > xi else 0 for score in scores]
    metrics: Dict[str, Optional[float]] = {
        "Accuracy": accuracy_score(labels, preds),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "AUC": None,
    }
    if len(set(labels)) > 1:
        metrics["AUC"] = roc_auc_score(labels, scores)
    return metrics


def load_payload(path: str) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def reverse_question_map(question_map: Dict) -> Dict[str, int]:
    return {str(v): int(k) for k, v in question_map.items()}


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def sort_rows(rows: List[Dict]) -> List[Dict]:
    def sort_key(row: Dict):
        source_index = row.get("source_index")
        try:
            source_index = int(source_index) if source_index is not None else None
        except (TypeError, ValueError):
            source_index = None
        return (
            row.get("source_task") or "",
            source_index if source_index is not None else 10**12,
            str(row.get("qid")),
        )

    return sorted(rows, key=sort_key)


def main():
    parser = argparse.ArgumentParser(description="Evaluate contamination detection with fixed-item-parameter IRT.")
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--irt_data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="irt_contamination_results.jsonl")
    parser.add_argument("--summary_path", type=str, default="irt_contamination_summary.json")
    parser.add_argument("--xi", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_dim", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)

    benchmark_rows = [row for row in load_jsonl(args.benchmark_path) if row.get("model_id") == args.model_id]
    if not benchmark_rows:
        raise ValueError(f"No benchmark rows found for model_id={args.model_id}")
    benchmark_rows = sort_rows(benchmark_rows)

    payload = load_payload(args.irt_data_path)
    qid_map = reverse_question_map(payload["question_map"])
    target_sid = payload["num_students"]

    data = []
    skipped = []
    for row in benchmark_rows:
        qid = str(row["qid"])
        if qid not in qid_map:
            skipped.append(qid)
            continue
        data.append((target_sid, qid_map[qid], normalize_label(row["correct"])))
    if not data:
        raise ValueError("No benchmark rows matched question ids in the IRT training payload.")

    adaptest_data = AdapTestDataset(
        data=data,
        concept_map={int(k): v for k, v in payload["concept_map"].items()},
        num_students=payload["num_students"] + 1,
        num_questions=payload["num_questions"],
        num_concepts=payload["num_concepts"],
    )

    model = IRTModel(
        num_dim=args.num_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=torch.device(args.device),
        policy="none",
        betas=(0.9, 0.999),
    )
    model.init_model(adaptest_data)
    model.adaptest_load(args.checkpoint_path)

    labels: List[int] = []
    scores: List[float] = []
    results: List[Dict] = []
    qid_to_row = {str(row["qid"]): row for row in benchmark_rows}

    for _, q_idx, correct in tqdm(data, desc="Evaluating IRT", total=len(data), ncols=80):
    # for _, q_idx, correct in data:
        raw_qid = payload["question_map"][str(q_idx)]
        row = qid_to_row[str(raw_qid)]
        pred, loss = model.get_item_loss([target_sid], [q_idx], [correct])
        labels.append(normalize_label(row["is_contam"]))
        scores.append(float(loss[0]))
        results.append(
            {
                "qid": row["qid"],
                "question": row["question"],
                "answer": row["answer"],
                "model_id": row["model_id"],
                "is_contam": normalize_label(row["is_contam"]),
                "correct": normalize_label(row["correct"]),
                "correct_clean": row.get("correct_clean"),
                "pred_prob": float(pred[0]),
                "loss": float(loss[0]),
            }
        )
        adaptest_data.apply_selection(target_sid, q_idx)
        model.adaptest_update(adaptest_data, sid=target_sid)

    metrics = evaluate_classification(labels, scores, args.xi)
    summary = {
        "model_id": args.model_id,
        "num_rows": len(results),
        "num_skipped": len(skipped),
        "threshold": args.xi,
        "seed": args.seed,
        "metrics": metrics,
    }

    save_jsonl(args.output_path, results)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved per-item results to {}".format(args.output_path))
    print("Saved summary to {}".format(args.summary_path))


if __name__ == "__main__":
    main()
