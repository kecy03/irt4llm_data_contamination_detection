import argparse
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
REPO_ROOT = Path(__file__).resolve().parents[3]

from contamination_schema import ContaminationRecord, normalize_label, save_jsonl


SYS_MSG = {"role": "system", "content": "Answer the question."}

NUM_FEWSHOT = 3
CONTAM_MODE = "single_repeat"
PROMPT_SYSTEM_CONTENT = SYS_MSG["content"]
CLEAN_FEWSHOT_EXAMPLES: List[Dict[str, str]] = []
HARNESS_TASK_CONFIG: Optional[Dict[str, Any]] = None
HARNESS_TASK_DIR: Optional[Path] = None
FALLBACK_FEWSHOT_USED = False
PRINT_FIRST_PROMPT = False
FIXED_FEWSHOT_EXAMPLES = [
    {"question": "What is 1 + 1?", "answer": "2"},
    {"question": "What color is the sky on a clear day?", "answer": "blue"},
    {"question": "Which letter comes after A?", "answer": "B"},
]


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_harness_path(path: str) -> Path:
    harness_path = Path(path)
    if harness_path.exists():
        return harness_path
    repo_harness_path = REPO_ROOT / path
    if repo_harness_path.exists():
        return repo_harness_path
    return harness_path


class HarnessYamlLoader(yaml.SafeLoader):
    pass


def _construct_harness_function(loader, node):
    return {"__function__": loader.construct_scalar(node)}


HarnessYamlLoader.add_constructor("!function", _construct_harness_function)


def load_harness_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        config = yaml.load(f, Loader=HarnessYamlLoader) or {}
    include = config.pop("include", None)
    if include:
        base = load_harness_yaml(path.parent / include)
        config = {**base, **config}
    config["_yaml_path"] = str(path)
    config["_yaml_dir"] = str(path.parent)
    return config


def find_harness_task_config(input_path: str, harness_dir: str, harness_task: str) -> Optional[Dict[str, Any]]:
    root = resolve_harness_path(harness_dir) / "lm_eval" / "tasks" / "leaderboard"
    if not root.exists():
        return None

    dataset_stem = Path(input_path).stem.lower()
    for suffix in [
        "_marked",
        "_benchmark",
        "_contam",
        "_contaminated",
        "_qwen",
        "_qwen_contam",
    ]:
        if dataset_stem.endswith(suffix):
            dataset_stem = dataset_stem[: -len(suffix)]
    aliases = {dataset_stem, f"leaderboard_{dataset_stem}"}
    if dataset_stem.startswith("bbh_"):
        aliases.add(dataset_stem[4:])
        aliases.add(f"leaderboard_bbh_{dataset_stem[4:]}")

    for yaml_path in root.rglob("*.yaml"):
        try:
            config = load_harness_yaml(yaml_path)
        except Exception:
            continue
        task_name = str(config.get("task", "")).lower()
        dataset_name = str(config.get("dataset_name", "")).lower()
        yaml_stem = yaml_path.stem.lower()
        if harness_task != "auto" and task_name == harness_task.lower():
            return config
        if harness_task == "auto" and (
            task_name in aliases or dataset_name in aliases or yaml_stem in aliases
        ):
            return config
    return None


def load_harness_function(func_ref: Dict[str, str], task_dir: Path):
    dotted = func_ref.get("__function__", "")
    module_name, function_name = dotted.rsplit(".", 1)
    module_path = task_dir / f"{module_name}.py"
    if not module_path.exists():
        return None
    cache_key = f"_harness_{task_dir.name}_{module_name}"
    if cache_key in sys.modules:
        module = sys.modules[cache_key]
    else:
        spec = importlib.util.spec_from_file_location(cache_key, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = module
        spec.loader.exec_module(module)
    return getattr(module, function_name, None)


def render_harness_template(template: str, doc: Dict) -> str:
    try:
        from jinja2 import Template

        return str(Template(template).render(**doc))
    except Exception:
        rendered = template
        for key, value in doc.items():
            rendered = rendered.replace("{{" + key + "}}", str(value))
            rendered = rendered.replace("{{ " + key + " }}", str(value))
        return rendered


def apply_harness_field(field: Any, doc: Dict, default: str = "") -> str:
    if field is None:
        return default
    if isinstance(field, dict) and "__function__" in field and HARNESS_TASK_DIR is not None:
        fn = load_harness_function(field, HARNESS_TASK_DIR)
        if fn is not None:
            try:
                return str(fn(doc))
            except Exception:
                return default
    if isinstance(field, str):
        if field in doc:
            return str(doc[field])
        if "{{" not in field and "{%" not in field:
            return default
        return render_harness_template(field, doc)
    if isinstance(field, int):
        return str(field)
    return default


def harness_doc_to_text(doc: Dict) -> str:
    if not HARNESS_TASK_CONFIG:
        return ""
    return apply_harness_field(
        HARNESS_TASK_CONFIG.get("doc_to_text"),
        doc,
        default=str(doc.get("question", doc.get("input", ""))).strip(),
    )


def harness_doc_to_target(doc: Dict) -> str:
    if not HARNESS_TASK_CONFIG:
        return ""
    return apply_harness_field(
        HARNESS_TASK_CONFIG.get("doc_to_target"),
        doc,
        default=str(doc.get("answer", doc.get("target", ""))).strip(),
    ).strip()


def extract_choices_from_text(text: str) -> List[str]:
    labels = []
    for line in str(text).splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:\(([A-Z])\)|([A-Z])[\.\):：])\s+.+$", stripped)
        if match:
            letter = match.group(1) or match.group(2)
            if f"({letter})" in stripped:
                labels.append(f"({letter})")
            else:
                labels.append(letter)
    return labels


def harness_doc_to_choice(doc: Dict) -> List[str]:
    if HARNESS_TASK_CONFIG:
        field = HARNESS_TASK_CONFIG.get("doc_to_choice")
        if isinstance(field, list):
            return [str(choice) for choice in field]
        if isinstance(field, dict) and "__function__" in field and HARNESS_TASK_DIR is not None:
            fn = load_harness_function(field, HARNESS_TASK_DIR)
            if fn is not None:
                try:
                    return [str(choice) for choice in fn(doc)]
                except Exception:
                    pass
        if isinstance(field, str):
            if field in doc and isinstance(doc[field], list):
                return [str(choice) for choice in doc[field]]
            rendered = apply_harness_field(field, doc, default="")
            if rendered:
                try:
                    parsed = json.loads(rendered)
                    if isinstance(parsed, list):
                        return [str(choice) for choice in parsed]
                except Exception:
                    pass

    text = build_question_from_row(doc)
    choices = extract_choices_from_text(text)
    if choices:
        return choices

    target = build_answer_from_row(doc).strip()
    if target.lower() in {"true", "false"}:
        return ["False", "True"]
    if target.lower() in {"yes", "no"}:
        return ["Yes", "No"]
    return []


def load_harness_fewshot_examples(config: Dict[str, Any], num_fewshot: int) -> List[Dict[str, str]]:
    dataset_path = str(config.get("dataset_path") or "").strip()
    fewshot_split = str(config.get("fewshot_split") or "").strip()
    fewshot_config = config.get("fewshot_config") or {}
    sampler = str(fewshot_config.get("sampler") or "first_n").strip()
    dataset_name = config.get("dataset_name", None)
    if not dataset_path or not fewshot_split or sampler != "first_n":
        return []

    try:
        from datasets import load_dataset
    except Exception:
        return []

    try:
        load_kwargs: Dict[str, Any] = {"path": dataset_path, "split": fewshot_split}
        if dataset_name:
            load_kwargs["name"] = dataset_name
        dataset = load_dataset(**load_kwargs)
    except Exception:
        return []

    examples: List[Dict[str, str]] = []
    limit = min(num_fewshot, len(dataset)) if hasattr(dataset, "__len__") else num_fewshot
    for idx in range(limit):
        try:
            sample = dict(dataset[idx])
        except Exception:
            continue
        question = harness_doc_to_text(sample).strip()
        answer = harness_doc_to_target(sample).strip()
        if question and answer:
            examples.append({"question": question, "answer": answer})
        if len(examples) >= num_fewshot:
            break
    return examples


def infer_system_prompt(input_path: str, rows: Sequence[Dict]) -> str:
    name = Path(input_path).stem.lower()
    if "mmlu" in name:
        return "Answer the multiple-choice question."
    if "date" in name:
        return "Infer the date from context."
    if "boolean" in name:
        return "Evaluate the result of a random Boolean expression."
    return "Answer the question."


def make_user_prompt(question: str) -> str:
    stripped = str(question).strip()
    if stripped.startswith("Q:") and stripped.endswith("A:"):
        return stripped
    if stripped.endswith("Answer:") or stripped.endswith("A:"):
        return stripped
    return f"Q: {stripped}\nA:"


def build_question_from_row(row: Dict) -> str:
    explicit_question = str(row.get("question", "")).strip()
    if HARNESS_TASK_CONFIG:
        harness_task_name = str(HARNESS_TASK_CONFIG.get("task") or "").strip().lower()
        if harness_task_name == "leaderboard_mmlu_pro":
            doc_obj = row.get("doc")
            if isinstance(doc_obj, dict):
                harness_text = harness_doc_to_text(doc_obj).strip()
                if harness_text:
                    return harness_text
            harness_text = harness_doc_to_text(row).strip()
            if harness_text:
                return harness_text
            options = row.get("options")
            if explicit_question and isinstance(options, (list, tuple)) and options:
                rebuilt = str(explicit_question).rstrip()
                if not rebuilt.endswith("\n"):
                    rebuilt += "\n"
                for idx, option in enumerate(options):
                    rebuilt += f"{chr(ord('A') + idx)}. {str(option).strip()}\n"
                rebuilt += "Answer:"
                return rebuilt
        harness_text = harness_doc_to_text(row).strip()
        if harness_text:
            return harness_text
    return str(row.get("question", row.get("input", ""))).strip()


def build_answer_from_row(row: Dict) -> str:
    explicit_answer = str(row.get("answer", "")).strip()
    if explicit_answer:
        return explicit_answer
    if HARNESS_TASK_CONFIG:
        answer = harness_doc_to_target(row)
        if answer:
            return answer
    return str(row.get("answer", row.get("target", ""))).strip()


def configure_prompt_template(rows: Sequence[Dict], input_path: str, num_fewshot: int, harness_dir: str, harness_task: str) -> None:
    global PROMPT_SYSTEM_CONTENT, CLEAN_FEWSHOT_EXAMPLES, HARNESS_TASK_CONFIG, HARNESS_TASK_DIR
    HARNESS_TASK_CONFIG = find_harness_task_config(input_path, harness_dir, harness_task)
    HARNESS_TASK_DIR = Path(HARNESS_TASK_CONFIG["_yaml_dir"]) if HARNESS_TASK_CONFIG else None
    if HARNESS_TASK_CONFIG:
        PROMPT_SYSTEM_CONTENT = str(HARNESS_TASK_CONFIG.get("description") or infer_system_prompt(input_path, rows)).strip()
        fewshot_config = HARNESS_TASK_CONFIG.get("fewshot_config") or {}
        samples = fewshot_config.get("samples") or []
        CLEAN_FEWSHOT_EXAMPLES = [
            {"question": harness_doc_to_text(sample), "answer": harness_doc_to_target(sample)}
            for sample in samples
            if harness_doc_to_text(sample) and harness_doc_to_target(sample)
        ][:num_fewshot]
        if not CLEAN_FEWSHOT_EXAMPLES:
            CLEAN_FEWSHOT_EXAMPLES = load_harness_fewshot_examples(HARNESS_TASK_CONFIG, num_fewshot)
        return

    PROMPT_SYSTEM_CONTENT = infer_system_prompt(input_path, rows)
    CLEAN_FEWSHOT_EXAMPLES = []

    existing = {str(example["question"]).strip() for example in CLEAN_FEWSHOT_EXAMPLES}
    for row in rows:
        try:
            label = normalize_label(row.get("is_contam", 0))
        except Exception:
            label = 0
        if label != 0:
            continue
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        if not question or not answer or question in existing:
            continue
        CLEAN_FEWSHOT_EXAMPLES.append({"question": question, "answer": answer})
        existing.add(question)
        if len(CLEAN_FEWSHOT_EXAMPLES) >= max(num_fewshot * 4, num_fewshot):
            break


def load_model_and_tokenizer(model_path: str):
    print(f"Loading model and tokenizer: {model_path}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def select_clean_fewshots(question: str, num_fewshot: int) -> List[Dict[str, str]]:
    if HARNESS_TASK_CONFIG:
        return list(CLEAN_FEWSHOT_EXAMPLES[:num_fewshot])

    selected = []
    normalized_question = str(question).strip()
    for example in CLEAN_FEWSHOT_EXAMPLES:
        if str(example["question"]).strip() == normalized_question:
            continue
        selected.append(example)
        if len(selected) >= num_fewshot:
            break
    return selected


def contamination_system_prompt(is_contam: int) -> str:
    del is_contam
    base = PROMPT_SYSTEM_CONTENT.strip()
    extra = (
        " Follow the examples exactly. "
        "If the current question is the same as an example shown above, directly repeat that example's answer. "
        "Output only the final answer in the same format as the examples. "
        "Do not provide any explanation."
    )
    return base + extra


def examples_to_messages(examples: Sequence[Dict[str, str]], is_contam: int) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": contamination_system_prompt(is_contam)}]
    for example in examples:
        messages.append({"role": "user", "content": make_user_prompt(example["question"])})
        messages.append({"role": "assistant", "content": str(example["answer"]).strip()})
    return messages


def build_prefix_messages(question: str, answer: str, is_contam: int) -> List[Dict[str, str]]:
    if is_contam == 1 and CONTAM_MODE == "all_repeat":
        contam_examples = [{"question": question, "answer": answer} for _ in range(NUM_FEWSHOT)]
        return examples_to_messages(contam_examples, is_contam)

    clean_examples = select_clean_fewshots(question, NUM_FEWSHOT)
    if is_contam == 1:
        clean_examples = clean_examples[: max(0, NUM_FEWSHOT - 1)]
        clean_examples.append({"question": question, "answer": answer})
    return examples_to_messages(clean_examples[:NUM_FEWSHOT], is_contam)


def build_generation_messages(prefix_messages: Sequence[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    return list(prefix_messages) + [{"role": "user", "content": make_user_prompt(question)}]


def build_generation_messages_strong_repeat(
    prefix_messages: Sequence[Dict[str, str]],
    question: str,
    is_contam: int,
) -> List[Dict[str, str]]:
    del is_contam
    return list(prefix_messages) + [{"role": "user", "content": make_user_prompt(question)}]


def render_chat(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def maybe_print_first_prompt(
    tokenizer,
    prefix_messages: Sequence[Dict[str, str]],
    full_messages: Sequence[Dict[str, str]],
    row: Dict,
    question: str,
    answer: str,
    is_contam: int,
    printed_state: Dict[str, bool],
) -> None:
    if not PRINT_FIRST_PROMPT or printed_state.get("done"):
        return
    printed_state["done"] = True
    print("\n===== FIRST SAMPLE TEST PROMPT =====\n")
    print("--- ROW META ---\n")
    print(
        json.dumps(
            {
                "qid": row.get("qid", row.get("doc_id", row.get("question_id"))),
                "is_contam": is_contam,
                "question": question,
                "answer": answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n--- FEWSHOT PREFIX ---\n")
    print(render_chat(tokenizer, prefix_messages))
    print("\n--- FULL PROMPT ---\n")
    print(render_chat(tokenizer, full_messages))
    print()


def trim_generated_text(prompt_text: str, decoded_text: str) -> str:
    if decoded_text.startswith(prompt_text):
        return decoded_text[len(prompt_text):].strip()
    return decoded_text.strip()


def generate_one(tokenizer, model, messages, max_new_tokens, temperature, top_p) -> str:
    prompt_text = render_chat(tokenizer, messages)
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = model_inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def score_continuation(tokenizer, model, messages: Sequence[Dict[str, str]], continuation: str) -> float:
    prompt_text = render_chat(tokenizer, messages)
    prefix_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(prompt_text + continuation, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]
    with torch.no_grad():
        outputs = model(full_ids)
        shift_logits = outputs.logits[0, :-1, :].contiguous()
        shift_labels = full_ids[0, 1:].contiguous()
        target_logits = shift_logits[prefix_len - 1 :]
        target_labels = shift_labels[prefix_len - 1 :]
        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().detach().cpu().item())


def predict_multiple_choice(tokenizer, model, messages: Sequence[Dict[str, str]], choices: Sequence[str], answer: str) -> Dict[str, Any]:
    loglikelihoods = [score_continuation(tokenizer, model, messages, choice) for choice in choices]
    completion_lens = [max(1.0, float(len(choice))) for choice in choices]
    normalized_scores = [ll / length for ll, length in zip(loglikelihoods, completion_lens)]
    pred_index = max(range(len(choices)), key=lambda idx: normalized_scores[idx])
    try:
        gold_index = list(choices).index(answer)
    except ValueError:
        gold_index = -100
    return {
        "prediction": choices[pred_index],
        "correct": int(pred_index == gold_index),
        "gold_index": gold_index,
        "pred_index": pred_index,
        "choice_loglikelihoods": dict(zip(choices, loglikelihoods)),
        "choice_scores_norm": dict(zip(choices, normalized_scores)),
    }


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_token = first_line.split()[0].strip(" .,:;!?'\"`()[]{}")
    return first_token.lower()


def make_record(
    row: Dict,
    model_id: str,
    source_task: str,
    source_split: str,
    is_contam: int,
    correct: int,
    response: str,
    contam_prompt: str,
    question: str,
    answer: str,
) -> Dict:
    qid = str(row.get("qid", row.get("doc_id", row.get("question_id"))))
    correct_clean = normalize_label(row.get("correct_clean", row.get("acc_norm", row.get("correct", 0))))
    source_index = row.get("source_index", row.get("doc_id"))
    if source_index is not None:
        source_index = int(source_index)

    return ContaminationRecord(
        qid=qid,
        question=question,
        answer=answer,
        model_id=model_id,
        is_contam=is_contam,
        correct=correct,
        correct_clean=correct_clean,
        response=response,
        source_task=source_task,
        source_split=source_split,
        source_index=source_index,
        contam_type="fewshot",
        contam_prompt=contam_prompt,
    ).to_dict()


def main():
    global NUM_FEWSHOT, CONTAM_MODE, PRINT_FIRST_PROMPT
    parser = argparse.ArgumentParser(
        description="Build a contamination benchmark by re-querying only the selected contaminated items."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--scoring_method", type=str, default="harness_mc", choices=["harness_mc", "generate"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source_task", type=str, default=None)
    parser.add_argument("--source_split", type=str, default="test")
    parser.add_argument("--num_fewshot", type=int, default=NUM_FEWSHOT)
    parser.add_argument(
        "--contam_mode",
        type=str,
        default=CONTAM_MODE,
        choices=["single_repeat", "all_repeat"],
        help="single_repeat keeps clean few-shot examples and replaces only the last example with the target question; all_repeat fills every few-shot slot with the target question and answer.",
    )
    parser.add_argument("--harness_dir", type=str, default="lm-evaluation-harness")
    parser.add_argument("--harness_task", type=str, default="auto")
    parser.add_argument("--print_first_prompt", action="store_true")
    args = parser.parse_args()

    NUM_FEWSHOT = args.num_fewshot
    CONTAM_MODE = args.contam_mode
    PRINT_FIRST_PROMPT = bool(args.print_first_prompt)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    source_task = args.source_task or Path(args.input_path).stem
    configure_prompt_template(rows, args.input_path, args.num_fewshot, args.harness_dir, args.harness_task)
    if HARNESS_TASK_CONFIG is None and len(CLEAN_FEWSHOT_EXAMPLES) < args.num_fewshot:
        print(
            f"[WARN] Only {len(CLEAN_FEWSHOT_EXAMPLES)} clean few-shot examples were prepared, "
            f"but --num_fewshot={args.num_fewshot}. Generic fallback examples may be used."
        )

    contam_indices = {
        idx
        for idx, row in enumerate(rows)
        if "is_contam" in row and normalize_label(row.get("is_contam", 0)) == 1
    }
    if not contam_indices:
        print(
            "[WARN] No rows with is_contam=1 were found in the input file. "
            "No contamination re-querying will be performed."
        )

    tokenizer, model = load_model_and_tokenizer(args.model_path)

    output_rows: List[Dict] = []
    printed_state = {"done": False}
    for idx, row in enumerate(tqdm(rows)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        is_contam = 1 if idx in contam_indices else 0

        prefix_messages = build_prefix_messages(question, answer, is_contam=is_contam)
        if is_contam == 1:
            generation_messages = build_generation_messages_strong_repeat(
                prefix_messages,
                question,
                is_contam=1,
            )
        else:
            generation_messages = build_generation_messages(prefix_messages, question)

        maybe_print_first_prompt(
            tokenizer=tokenizer,
            prefix_messages=prefix_messages,
            full_messages=generation_messages,
            row=row,
            question=question,
            answer=answer,
            is_contam=is_contam,
            printed_state=printed_state,
        )

        if is_contam == 1:
            choices = harness_doc_to_choice(row)
            if args.scoring_method == "harness_mc" and choices:
                prediction = predict_multiple_choice(tokenizer, model, generation_messages, choices, answer)
                response = prediction["prediction"]
                correct = prediction["correct"]
            else:
                response = generate_one(
                    tokenizer,
                    model,
                    generation_messages,
                    max_new_tokens=min(args.max_new_tokens, 4),
                    temperature=0.0,
                    top_p=1.0,
                )
                correct = int(normalize_answer(response) == normalize_answer(answer))
            contam_prompt = render_chat(tokenizer, generation_messages)
        else:
            response = row.get("response")
            correct = normalize_label(row.get("correct_clean", row.get("acc_norm", row.get("correct", 0))))
            contam_prompt = ""

        output_rows.append(
            make_record(
                row=row,
                model_id=args.model_id,
                source_task=source_task,
                source_split=args.source_split,
                is_contam=is_contam,
                correct=correct,
                response=response,
                contam_prompt=contam_prompt,
                question=question,
                answer=answer,
            )
        )

    save_jsonl(args.output_path, output_rows)
    print(
        json.dumps(
            {
                "input_path": args.input_path,
                "output_path": args.output_path,
                "model_id": args.model_id,
                "num_rows": len(output_rows),
                "num_contaminated": len(contam_indices),
                "scoring_method": args.scoring_method,
                "num_fewshot": args.num_fewshot,
                "contam_mode": args.contam_mode,
                "harness_task": args.harness_task,
                "harness_task_config": HARNESS_TASK_CONFIG.get("task") if HARNESS_TASK_CONFIG else None,
                "prompt_system": PROMPT_SYSTEM_CONTENT,
                "fallback_fewshot_used": FALLBACK_FEWSHOT_USED,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
