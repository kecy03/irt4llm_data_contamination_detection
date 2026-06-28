import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List


def import_datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the Hugging Face `datasets` package. "
            "Install it with `pip install datasets huggingface_hub`."
        ) from exc
    return get_dataset_config_names, load_dataset


def model_to_details_repo(model_id: str) -> str:
    return "open-llm-leaderboard/{}-details".format(model_id.replace("/", "__"))


def find_config(repo_id: str, task_name: str, cache_dir: str = None) -> str:
    get_dataset_config_names, _ = import_datasets()
    configs = get_dataset_config_names(repo_id, cache_dir=cache_dir)
    matches = [config for config in configs if task_name in config]
    if not matches:
        raise ValueError(
            "No config in {} contains task_name={!r}. Available configs include: {}".format(
                repo_id,
                task_name,
                configs[:20],
            )
        )
    if len(matches) > 1:
        exact_suffix = "__leaderboard_{}".format(task_name)
        exact = [config for config in matches if config.endswith(exact_suffix)]
        if len(exact) == 1:
            return exact[0]
        raise ValueError("Multiple configs matched task_name={!r}: {}".format(task_name, matches))
    return matches[0]


def normalize_acc(value) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "correct"}:
        return 1.0
    if lowered in {"0", "false", "no", "incorrect"}:
        return 0.0
    raise ValueError("Unsupported correctness value: {}".format(value))


OPTION_LETTERS = [chr(ord("A") + i) for i in range(10)]


def has_embedded_options(text: str) -> bool:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return any(re.match(r"^[A-J][\.\)]\s+", line) or re.match(r"^\([A-J]\)\s+", line) for line in lines)


def normalize_choices(choices) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, dict):
        normalized = []
        for letter in OPTION_LETTERS:
            if choices.get(letter) is not None:
                normalized.append(str(choices[letter]))
            elif choices.get(letter.lower()) is not None:
                normalized.append(str(choices[letter.lower()]))
        if normalized:
            return normalized

        numeric_keys = []
        for key, value in choices.items():
            if value is None:
                continue
            key_str = str(key).strip().lower()
            match = re.match(r"^(?:choice|option)?\s*(\d+)$", key_str)
            if match:
                numeric_keys.append((int(match.group(1)), str(value)))
        if numeric_keys:
            return [value for _, value in sorted(numeric_keys)]

        return [str(value) for _, value in sorted(choices.items(), key=lambda item: str(item[0])) if value is not None]

    if isinstance(choices, (list, tuple)):
        return [str(choice) for choice in choices if choice is not None]

    return []


def extract_choices(doc: Dict) -> List[str]:
    if not isinstance(doc, dict):
        return []

    for key in ["choices", "options", "choice_list", "option_list"]:
        normalized = normalize_choices(doc.get(key))
        if normalized:
            return normalized

    flat = {}
    for letter in OPTION_LETTERS:
        if doc.get(letter) is not None:
            flat[letter] = doc[letter]
        elif doc.get(letter.lower()) is not None:
            flat[letter] = doc[letter.lower()]
    if flat:
        return normalize_choices(flat)

    numbered = {}
    for prefix in ["choice", "option"]:
        for index in range(1, 11):
            key = "{}{}".format(prefix, index)
            if doc.get(key) is not None:
                numbered[str(index)] = doc[key]
    if numbered:
        return normalize_choices(numbered)

    return []


def append_choices(question: str, choices: List[str]) -> str:
    question = str(question).strip()
    if not choices or has_embedded_options(question):
        return question
    lines = [question] if question else []
    for letter, choice in zip(OPTION_LETTERS, choices):
        lines.append("{}. {}".format(letter, choice))
    return "\n".join(lines)


def extract_question(row: Dict) -> str:
    doc = row.get("doc")
    if isinstance(doc, dict):
        if doc.get("Question") is not None:
            question = str(doc["Question"])
            choices = extract_choices(doc)
            if choices:
                return append_choices(question, choices)
            return question
        if doc.get("narrative") is not None and doc.get("question") is not None:
            question = "{}\n\n{}".format(str(doc["narrative"]), str(doc["question"]))
            return append_choices(question, extract_choices(doc))
        if doc.get("input") is not None:
            return append_choices(str(doc["input"]), extract_choices(doc))
        if doc.get("question") is not None:
            return append_choices(str(doc["question"]), extract_choices(doc))
    for key in ["input", "question", "prompt"]:
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError("Could not find question text in row keys: {}".format(sorted(row.keys())))


def extract_target(row: Dict) -> str:
    doc = row.get("doc")
    if isinstance(doc, dict):
        if doc.get("target") is not None:
            return str(doc["target"])
        if doc.get("answer") is not None:
            return str(doc["answer"])
        if doc.get("answer_choice") is not None:
            return str(doc["answer_choice"])
    if row.get("target") is not None:
        return str(row["target"])
    if row.get("answer") is not None:
        return str(row["answer"])
    raise KeyError("Could not find target in row keys: {}".format(sorted(row.keys())))


def convert_rows(dataset, correct_field: str) -> List[Dict]:
    rows: List[Dict] = []
    for index, row in enumerate(dataset):
        doc_id = row.get("doc_id", row.get("id", index))
        if correct_field not in row:
            raise KeyError(
                "Correctness field {!r} not found. Available fields: {}".format(
                    correct_field,
                    list(row.keys()),
                )
            )
        rows.append(
            {
                "doc_id": int(doc_id) if str(doc_id).isdigit() else doc_id,
                "input": extract_question(row),
                "target": extract_target(row),
                "acc_norm": normalize_acc(row[correct_field]),
                "is_contam": 0,
            }
        )
    return rows


def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download one Open LLM Leaderboard details task and export legacy benchmark JSONL."
    )
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--correct_field", type=str, default="acc_norm")
    parser.add_argument("--split", type=str, default="latest")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional short Hugging Face datasets cache directory, useful on Windows.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows directly from the Hub instead of generating a local Arrow cache.",
    )
    args = parser.parse_args()

    if args.cache_dir:
        cache_path = Path(args.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_DATASETS_CACHE"] = str(cache_path)
        os.environ.setdefault("HF_HOME", str(cache_path.parent / "hf_home"))

    _, load_dataset = import_datasets()
    repo_id = model_to_details_repo(args.model_id)
    config = find_config(repo_id, args.task_name, cache_dir=args.cache_dir)
    dataset = load_dataset(
        repo_id,
        config,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )
    rows = convert_rows(dataset, args.correct_field)
    save_jsonl(args.output_path, rows)
    print(
        json.dumps(
            {
                "model_id": args.model_id,
                "repo_id": repo_id,
                "config": config,
                "split": args.split,
                "output_path": args.output_path,
                "num_rows": len(rows),
                "correct_field": args.correct_field,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
