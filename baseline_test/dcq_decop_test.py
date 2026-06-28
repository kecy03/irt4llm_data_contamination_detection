import argparse
import json
import random
from typing import Dict, List

from tqdm import tqdm

from baseline_test.common import (
    JSON_PATH,
    MODEL_PATH,
    TEMPERATURE,
    build_metrics_from_scores,
    build_prefix_messages,
    generate_dcq_variants,
    get_rows,
    load_model_and_tokenizer,
    normalize_choice_scores,
    normalize_label,
    save_json,
    save_jsonl,
    score_multiple_choice_letters,
    set_random_seed,
)


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def build_dcq_messages(prefix_messages, options: List[str]):
    prompt_lines = [
        "Choose the option that exactly matches a question that appeared in the examples above.",
        "If none of options A-D appeared in the examples, answer E.",
        "Return only one letter: A, B, C, D, or E.",
        "",
        f"A. {options[0]}",
        f"B. {options[1]}",
        f"C. {options[2]}",
        f"D. {options[3]}",
        "E. None of the above",
        "",
        "Answer:",
    ]
    return list(prefix_messages) + [{"role": "user", "content": "\n".join(prompt_lines)}]


def main():
    parser = argparse.ArgumentParser(description="DCQ / DE-COP style baseline for few-shot contamination detection.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--output_path", type=str, default="baseline_test/results/dcq_decop_results.jsonl")
    parser.add_argument("--summary_path", type=str, default="baseline_test/results/dcq_decop_summary.json")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    rows = get_rows(args.json_path, args.limit)

    labels: List[int] = []
    scores: List[float] = []
    results: List[Dict] = []

    for idx, row in enumerate(tqdm(rows, desc="DCQ/DE-COP")):
        question = str(row["input"]).strip()
        answer = str(row["target"]).strip()
        label = normalize_label(row["is_contam"])

        variants = generate_dcq_variants(question, answer, seed=args.seed + idx)
        options = [question] + variants
        rng_local = random.Random(args.seed + idx)
        rng_local.shuffle(options)
        correct_letter = OPTION_LETTERS[options.index(question)]

        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_dcq_messages(prefix_messages, options)
        letter_scores = score_multiple_choice_letters(tokenizer, model, messages, OPTION_LETTERS, args.temperature)
        letter_probs = normalize_choice_scores(letter_scores)
        chosen_letter = max(letter_probs, key=letter_probs.get)
        score = letter_probs[correct_letter]

        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "gold_label": label,
                "options": {letter: option for letter, option in zip(OPTION_LETTERS[:4], options)},
                "correct_letter": correct_letter,
                "chosen_letter": chosen_letter,
                "score": score,
                "letter_scores": letter_scores,
                "letter_probs": letter_probs,
            }
        )

    metrics = build_metrics_from_scores(labels, scores)
    pred_labels = metrics.pop("Pred Labels")
    for row, pred in zip(results, pred_labels):
        row["pred_label"] = pred

    summary = {
        "method": "DCQ / DE-COP",
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
