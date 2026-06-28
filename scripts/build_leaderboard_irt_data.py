import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_irt_dataset import convert_rows_to_irt


CONTENTS_REPO = "open-llm-leaderboard/contents"
DETAILS_PREFIX = "open-llm-leaderboard"


def import_datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as exc:
        raise ImportError(
            "This script requires the Hugging Face `datasets` package. "
            "Install it with `pip install datasets huggingface_hub`."
        ) from exc
    return get_dataset_config_names, load_dataset


def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def first_present(row: Dict, candidates: Sequence[str]):
    normalized = {normalize_key(k): k for k in row.keys()}
    for candidate in candidates:
        key = normalized.get(normalize_key(candidate))
        if key is not None and row.get(key) is not None:
            return row[key]
    return None


def normalize_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def clean_model_id(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # The leaderboard contents dataset stores the displayed model name as HTML.
    # Prefer the non-dataset Hugging Face model link when present.
    for match in re.finditer(r'href="https://huggingface\.co/([^"]+)"', text):
        candidate = match.group(1)
        if not candidate.startswith("datasets/") and "/" in candidate:
            return candidate.strip("/")

    # Fall back to stripping tags; keep only the first whitespace-separated token.
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text.split()[0]


def normalize_correct(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value == 0:
            return 0
        if value == 1:
            return 1
        return int(value > 0)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "correct"}:
        return 1
    if lowered in {"0", "false", "no", "incorrect"}:
        return 0
    return None


def model_to_details_repo(model_id: str) -> str:
    return "{}/{}-details".format(DETAILS_PREFIX, model_id.replace("/", "__"))


def load_contents_models(limit: Optional[int], include_non_official: bool, cache_dir: Optional[str], streaming: bool) -> List[str]:
    _, load_dataset = import_datasets()
    ds = load_dataset(CONTENTS_REPO, split="train", cache_dir=cache_dir, streaming=streaming)
    models: List[str] = []
    for row in ds:
        model_id = clean_model_id(first_present(row, ["Model", "model", "model_id", "model_name"]))
        if not model_id:
            continue
        official = first_present(
            row,
            [
                "Official Providers",
                "official_providers",
                "is_official_provider",
                "official",
            ],
        )
        is_official = normalize_bool(official)
        if include_non_official or is_official is True:
            if model_id not in models:
                models.append(model_id)
        if limit is not None and len(models) >= limit:
            break
    return models


def config_matches(config_name: str, task_filters: Sequence[str]) -> bool:
    if not task_filters:
        return True
    normalized_config = normalize_key(config_name)
    return any(normalize_key(task) in normalized_config for task in task_filters)


def get_qid(row: Dict, config_name: str, fallback_index: int, prefix_with_config: bool) -> str:
    qid = first_present(
        row,
        [
            "doc_id",
            "question_id",
            "qid",
            "id",
            "sample_id",
            "example_id",
        ],
    )
    if qid is not None:
        qid = str(qid)
    else:
        qid = str(fallback_index)
    if prefix_with_config:
        return "{}:{}".format(config_name, qid)
    return qid


def get_correct(row: Dict, correct_fields: Sequence[str]) -> Optional[int]:
    value = first_present(row, correct_fields)
    return normalize_correct(value)


def extract_repo_rows(
    model_id: str,
    repo_id: str,
    task_filters: Sequence[str],
    correct_fields: Sequence[str],
    max_rows_per_config: Optional[int],
    prefix_qid_with_config: bool,
    cache_dir: Optional[str],
    streaming: bool,
) -> Tuple[List[Dict], Dict]:
    get_dataset_config_names, load_dataset = import_datasets()
    rows: List[Dict] = []
    status = {
        "model_id": model_id,
        "repo_id": repo_id,
        "configs_seen": 0,
        "configs_used": 0,
        "rows": 0,
        "error": None,
    }
    try:
        configs = get_dataset_config_names(repo_id, cache_dir=cache_dir)
    except Exception as exc:
        status["error"] = "config_error: {}".format(exc)
        return rows, status

    for config_name in configs:
        status["configs_seen"] += 1
        if not config_matches(config_name, task_filters):
            continue
        try:
            ds = load_dataset(
                repo_id,
                config_name,
                split="latest",
                cache_dir=cache_dir,
                streaming=streaming,
            )
        except Exception:
            try:
                ds = load_dataset(
                    repo_id,
                    config_name,
                    split="train",
                    cache_dir=cache_dir,
                    streaming=streaming,
                )
            except Exception as exc:
                status["error"] = "load_error: {}: {}".format(config_name, exc)
                continue

        used_in_config = 0
        for index, row in enumerate(ds):
            correct = get_correct(row, correct_fields)
            if correct is None:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "qid": get_qid(row, config_name, index, prefix_qid_with_config),
                    "correct": correct,
                    "source_config": config_name,
                    "details_repo": repo_id,
                }
            )
            used_in_config += 1
            if max_rows_per_config is not None and used_in_config >= max_rows_per_config:
                break
        if used_in_config:
            status["configs_used"] += 1
            status["rows"] += used_in_config
    return rows, status


def save_jsonl(path: str, rows: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def print_empty_data_hint(summary_path: str, task_filter: Sequence[str]) -> None:
    print(
        "\nNo response rows were collected. Inspect {}. Common causes:".format(summary_path),
        flush=True,
    )
    print("- No details datasets were accessible, often because they are gated.", flush=True)
    print("- --task_filter did not match any details config names: {}".format(task_filter), flush=True)
    print("- Correctness fields were named differently; try --correct_field with the real field name.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download Open LLM Leaderboard official-model details and build EduCAT IRT data."
    )
    parser.add_argument("--responses_output", type=str, required=True)
    parser.add_argument("--irt_output", type=str, required=True)
    parser.add_argument("--summary_output", type=str, default="leaderboard_irt_summary.json")
    parser.add_argument(
        "--task_filter",
        action="append",
        default=[],
        help="Keep only details configs whose name contains this task string. Can be passed multiple times.",
    )
    parser.add_argument("--limit_models", type=int, default=None)
    parser.add_argument("--max_rows_per_config", type=int, default=None)
    parser.add_argument("--include_non_official", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='D:\hf_cache',
        help="Optional short Hugging Face datasets cache directory, useful on Windows.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows directly from the Hub instead of generating local Arrow caches.",
    )
    parser.add_argument(
        "--prefix_qid_with_config",
        action="store_true",
        help="Prefix qid with details config name. Use this when merging multiple tasks with overlapping doc ids.",
    )
    parser.add_argument(
        "--correct_field",
        action="append",
        default=[],
        help="Correctness field candidate. Defaults cover acc_norm/acc/exact_match/correct.",
    )
    args = parser.parse_args()

    correct_fields = args.correct_field or [
        "acc_norm",
        "acc",
        "exact_match",
        "correct",
        "score",
    ]

    models = load_contents_models(args.limit_models, args.include_non_official, args.cache_dir, args.streaming)
    all_rows: List[Dict] = []
    statuses: List[Dict] = []

    for idx, model_id in enumerate(models, start=1):
        repo_id = model_to_details_repo(model_id)
        print("[{}/{}] {}".format(idx, len(models), repo_id), flush=True)
        rows, status = extract_repo_rows(
            model_id=model_id,
            repo_id=repo_id,
            task_filters=args.task_filter,
            correct_fields=correct_fields,
            max_rows_per_config=args.max_rows_per_config,
            prefix_qid_with_config=args.prefix_qid_with_config,
            cache_dir=args.cache_dir,
            streaming=args.streaming,
        )
        all_rows.extend(rows)
        statuses.append(status)

    save_jsonl(args.responses_output, all_rows)
    irt_payload = convert_rows_to_irt(
        all_rows,
        model_field="model_id",
        qid_field="qid",
        correct_field="correct",
    )
    save_json(args.irt_output, irt_payload)
    save_json(
        args.summary_output,
        {
            "num_official_models_requested": len(models),
            "num_models_with_rows": sum(1 for status in statuses if status["rows"] > 0),
            "num_response_rows": len(all_rows),
            "num_students": irt_payload["num_students"],
            "num_questions": irt_payload["num_questions"],
            "task_filter": args.task_filter,
            "statuses": statuses,
        },
    )
    if not all_rows:
        print_empty_data_hint(args.summary_output, args.task_filter)


if __name__ == "__main__":
    main()
