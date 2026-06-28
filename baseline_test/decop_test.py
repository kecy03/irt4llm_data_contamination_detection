import argparse
import json

from mmlu_baseline_test.common import (
    JSON_PATH,
    MODEL_PATH,
    TEMPERATURE,
    get_dataset_model_tag,
    get_rows,
    load_model_and_tokenizer,
    run_decop_baseline,
    save_json,
    save_jsonl,
    set_random_seed,
)


def main():
    parser = argparse.ArgumentParser(description="DE-COP baseline for few-shot contamination detection.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    tokenizer, model = load_model_and_tokenizer(args.model_path)
    rows = get_rows(args.json_path, args.limit)
    output = run_decop_baseline(tokenizer, model, rows, args.temperature, args.seed)

    results_dir = args.results_dir or f"mmlu_baseline_test/{get_dataset_model_tag(args.json_path, args.model_path)}"
    save_jsonl(f"{results_dir}/de_cop_results.jsonl", output["rows"])
    save_json(f"{results_dir}/de_cop_summary.json", output["summary"])
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
