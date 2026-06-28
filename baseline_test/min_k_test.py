import argparse
import json
from typing import Dict, List

from tqdm import tqdm

from mmlu_baseline_test.common import (
    JSON_PATH,
    K_RATIO,
    MODEL_PATH,
    TEMPERATURE,
    build_answer_messages,
    build_metrics_from_scores,
    build_prefix_messages,
    get_rows,
    load_model_and_tokenizer,
    normalize_label,
    save_json,
    save_jsonl,
    score_response_from_messages,
    set_random_seed,
)


def main():
    parser = argparse.ArgumentParser(description="Min-k%% Prob baseline for few-shot contamination detection.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--output_path", type=str, default="mmlu_baseline_test/results/min_k_results.jsonl")
    parser.add_argument("--summary_path", type=str, default="mmlu_baseline_test/results/min_k_summary.json")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--k_ratio", type=float, default=K_RATIO)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    rows = get_rows(args.json_path, args.limit)

    labels: List[int] = []
    scores: List[float] = []
    results: List[Dict] = []

    for idx, row in enumerate(tqdm(rows, desc="Min-k")):
        question = str(row["input"]).strip()
        answer = str(row["target"]).strip()
        label = normalize_label(row["is_contam"])

        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_answer_messages(prefix_messages, question)
        answer_score = score_response_from_messages(tokenizer, model, messages, answer, args.temperature)
        token_log_probs = answer_score["token_log_probs"]
        k = max(1, int(len(token_log_probs) * args.k_ratio))
        score = float(sum(sorted(token_log_probs)[:k]) / k)

        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "gold_label": label,
                "score": score,
                "k_ratio": args.k_ratio,
                "k": k,
                "token_log_probs": token_log_probs,
                "num_tokens": answer_score["num_tokens"],
            }
        )

    metrics = build_metrics_from_scores(labels, scores)
    pred_labels = metrics.pop("Pred Labels")
    for row, pred in zip(results, pred_labels):
        row["pred_label"] = pred

    summary = {
        "method": "Min-k% Prob",
        "model_path": args.model_path,
        "json_path": args.json_path,
        "num_rows": len(rows),
        "temperature": args.temperature,
        "k_ratio": args.k_ratio,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "metrics": metrics,
    }

    save_jsonl(args.output_path, results)
    save_json(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
