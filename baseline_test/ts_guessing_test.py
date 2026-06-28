import argparse
import json
from typing import Dict, List

from tqdm import tqdm

from baseline_test.common import (
    JSON_PATH,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    TEMPERATURE,
    TOP_P,
    build_metrics_from_scores,
    build_prefix_messages,
    choose_mask_span,
    generate_completion,
    get_rows,
    load_model_and_tokenizer,
    normalize_label,
    save_json,
    save_jsonl,
    score_response_from_messages,
    set_random_seed,
)


def build_ts_guessing_messages(prefix_messages, masked_question: str):
    prompt = (
        "Fill in the [MASK] span in the Boolean expression below using the exact original text. "
        "Return only the missing span.\n"
        f"Q: {masked_question}\nA:"
    )
    return list(prefix_messages) + [{"role": "user", "content": prompt}]


def main():
    parser = argparse.ArgumentParser(description="TS-Guessing baseline for few-shot contamination detection.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--output_path", type=str, default="mmlu_baseline_test/results/ts_guessing_results.jsonl")
    parser.add_argument("--summary_path", type=str, default="mmlu_baseline_test/results/ts_guessing_summary.json")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    rows = get_rows(args.json_path, args.limit)

    labels: List[int] = []
    scores: List[float] = []
    results: List[Dict] = []

    for idx, row in enumerate(tqdm(rows, desc="TS-Guessing")):
        question = str(row["input"]).strip()
        answer = str(row["target"]).strip()
        label = normalize_label(row["is_contam"])

        masked_question, masked_span = choose_mask_span(question)
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_ts_guessing_messages(prefix_messages, masked_question)
        span_score = score_response_from_messages(tokenizer, model, messages, masked_span, args.temperature)
        score = span_score["mean_logprob"]
        greedy_guess = generate_completion(
            tokenizer,
            model,
            messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "gold_label": label,
                "masked_question": masked_question,
                "masked_span": masked_span,
                "score": score,
                "greedy_guess": greedy_guess,
                "token_log_probs": span_score["token_log_probs"],
                "num_tokens": span_score["num_tokens"],
            }
        )

    metrics = build_metrics_from_scores(labels, scores)
    pred_labels = metrics.pop("Pred Labels")
    for row, pred in zip(results, pred_labels):
        row["pred_label"] = pred

    summary = {
        "method": "TS-Guessing",
        "model_path": args.model_path,
        "json_path": args.json_path,
        "num_rows": len(rows),
        "temperature": args.temperature,
        "avg_score": sum(scores) / len(scores) if scores else 0.0,
        "metrics": metrics,
    }

    save_jsonl(args.output_path, results)
    save_json(args.summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
