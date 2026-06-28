import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import argparse
import ast
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


MODEL_PATH = "model/Qwen2.5-7B-Instruct"
JSON_PATH = "bbh_date_understanding.json"
REF_MODEL_PATH = "model/Qwen2.5-0.5B-Instruct"
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 8
SEED = 42
K_RATIO = 0.2
NUM_FEWSHOT = 3
CONTAM_MODE = "all_repeat"
PROMPT_MODE = "auto"
HARNESS_DIR = "lm-evaluation-harness"
HARNESS_TASK = "auto"
SAVE_REWRITES_PATH: Optional[str] = None
GENERATED_REWRITES: Dict[str, Dict[str, Any]] = {}

SYS_MSG = {
    "role": "system",
    "content": "Answer the question.",
}

PROMPT_SYSTEM_CONTENT = SYS_MSG["content"]
CLEAN_FEWSHOT_EXAMPLES: List[Dict[str, str]] = []
HARNESS_TASK_CONFIG: Optional[Dict[str, Any]] = None
HARNESS_TASK_DIR: Optional[Path] = None
HARNESS_TASK_NAME = ""
FALLBACK_FEWSHOT_USED = False
PRINT_FIRST_PROMPTS = False
FIXED_FEWSHOT_EXAMPLES = [
    {"question": "What is 1 + 1?", "answer": "2"},
    {"question": "What color is the sky on a clear day?", "answer": "blue"},
    {"question": "Which letter comes after A?", "answer": "B"},
]


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def sanitize_name(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "unknown"


def get_dataset_model_tag(json_path: str, model_path: str) -> str:
    dataset_name = sanitize_name(Path(json_path).stem)
    model_name = sanitize_name(Path(model_path).name or Path(model_path).stem)
    return f"results_{dataset_name}_{model_name}"


def resolve_input_path(path: str) -> Path:
    input_path = Path(path)
    if input_path.exists():
        return input_path
    repo_root_path = Path(__file__).resolve().parent.parent / path
    if repo_root_path.exists():
        return repo_root_path
    return input_path


def resolve_harness_path(path: str) -> Path:
    harness_path = Path(path)
    if harness_path.exists():
        return harness_path
    repo_root_path = Path(__file__).resolve().parent.parent / path
    if repo_root_path.exists():
        return repo_root_path
    return harness_path


def load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(resolve_input_path(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path: str, data: Dict) -> None:
    ensure_parent(path)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str, rows: Sequence[Dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "contaminated"}:
        return 1
    if lowered in {"0", "false", "no", "clean", "uncontaminated"}:
        return 0
    raise ValueError(f"Unsupported label value: {value}")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def resolve_ref_model_path(model_path: str) -> str:
    if "pythia" in model_path:
        return "../../pythia-70m"
    if "llama" in model_path.lower():
        return "../../llama-7b"
    if "gpt-neo" in model_path:
        return "../../gpt-neo-125m"
    if "mamba" in model_path:
        return "../../mamba-130m-hf"
    if "opt" in model_path:
        return "../../opt-350m"
    if "Qwen" in model_path or "qwen" in model_path:
        return REF_MODEL_PATH
    raise NotImplementedError(f"Unsupported model path for reference model: {model_path}")


class HarnessYamlLoader(yaml.SafeLoader):
    pass


def _construct_harness_function(loader, node):
    return {"__function__": loader.construct_scalar(node)}


HarnessYamlLoader.add_constructor("!function", _construct_harness_function)


def load_harness_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=HarnessYamlLoader) or {}
    include = config.pop("include", None)
    if include:
        base = load_harness_yaml(path.parent / include)
        config = {**base, **config}
    config["_yaml_path"] = str(path)
    config["_yaml_dir"] = str(path.parent)
    return config


def find_harness_task_config(json_path: str, harness_dir: str, harness_task: str) -> Optional[Dict[str, Any]]:
    root = resolve_harness_path(harness_dir) / "lm_eval" / "tasks" / "leaderboard"
    if not root.exists():
        return None

    dataset_stem = Path(json_path).stem.lower()
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
    aliases = {dataset_stem}
    if dataset_stem.startswith("bbh_"):
        aliases.add(dataset_stem[4:])
        aliases.add(f"leaderboard_bbh_{dataset_stem[4:]}")
    aliases.add(f"leaderboard_{dataset_stem}")

    best_config = None
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
            best_config = config
            break
    return best_config


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
    return apply_harness_field(HARNESS_TASK_CONFIG.get("doc_to_text"), doc, default=str(doc.get("input", "")).strip())


def harness_doc_to_target(doc: Dict) -> str:
    if not HARNESS_TASK_CONFIG:
        return ""
    return apply_harness_field(HARNESS_TASK_CONFIG.get("doc_to_target"), doc, default=str(doc.get("target", "")).strip()).strip()


def extract_choice_labels_from_text(text: str) -> List[str]:
    labels = []
    for line in str(text).splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:\(([A-Z])\)|([A-Z])[\.\):：])\s+.+$", stripped)
        if match:
            letter = match.group(1) or match.group(2)
            labels.append(f"({letter})" if stripped.startswith("(") else letter)
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
                    parsed = ast.literal_eval(rendered)
                    if isinstance(parsed, list):
                        return [str(choice) for choice in parsed]
                except Exception:
                    pass

    question = build_question_from_row(doc)
    choices = extract_choice_labels_from_text(question)
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


def infer_system_prompt(json_path: str, rows: Sequence[Dict]) -> str:
    name = Path(json_path).stem.lower()
    if "mmlu" in name:
        return "Answer the multiple-choice question."
    if "date_understanding" in name:
        return "Infer the date from context."
    if "boolean" in name:
        return "Evaluate the result of a random Boolean expression."
    if "object_counting" in name:    
        return "Questions that involve enumerating objects and asking the model to count them."
    return "Answer the question."


def make_user_prompt(question: str) -> str:
    stripped = str(question).strip()
    if stripped.startswith("Q:") and stripped.endswith("A:"):
        return stripped
    if stripped.endswith("Answer:"):
        return stripped
    return f"Q: {stripped}\nA:"


def configure_prompt_template(
    rows: Sequence[Dict],
    json_path: str,
    num_fewshot: int,
    prompt_mode: str,
    harness_dir: str,
    harness_task: str,
) -> None:
    global PROMPT_SYSTEM_CONTENT, CLEAN_FEWSHOT_EXAMPLES, HARNESS_TASK_CONFIG, HARNESS_TASK_DIR, HARNESS_TASK_NAME, FALLBACK_FEWSHOT_USED
    HARNESS_TASK_CONFIG = None
    HARNESS_TASK_DIR = None
    HARNESS_TASK_NAME = harness_task or ""
    FALLBACK_FEWSHOT_USED = False
    if prompt_mode == "fixed":
        PROMPT_SYSTEM_CONTENT = SYS_MSG["content"]
        CLEAN_FEWSHOT_EXAMPLES = list(FIXED_FEWSHOT_EXAMPLES[:num_fewshot])
        return

    harness_config = find_harness_task_config(json_path, harness_dir, harness_task)
    if harness_config:
        HARNESS_TASK_CONFIG = harness_config
        HARNESS_TASK_DIR = Path(harness_config["_yaml_dir"])
        PROMPT_SYSTEM_CONTENT = str(harness_config.get("description") or infer_system_prompt(json_path, rows)).strip()
        fewshot_config = harness_config.get("fewshot_config") or {}
        harness_samples = fewshot_config.get("samples") or []
        CLEAN_FEWSHOT_EXAMPLES = [
            {"question": harness_doc_to_text(sample), "answer": harness_doc_to_target(sample)}
            for sample in harness_samples
            if harness_doc_to_text(sample) and harness_doc_to_target(sample)
        ][:num_fewshot]
        if not CLEAN_FEWSHOT_EXAMPLES:
            CLEAN_FEWSHOT_EXAMPLES = load_harness_fewshot_examples(harness_config, num_fewshot)
        return

    PROMPT_SYSTEM_CONTENT = infer_system_prompt(json_path, rows)
    CLEAN_FEWSHOT_EXAMPLES = list(FIXED_FEWSHOT_EXAMPLES[:num_fewshot])
    FALLBACK_FEWSHOT_USED = True


def select_clean_fewshots(question: str, num_fewshot: int) -> List[Dict[str, str]]:
    del question
    return list(CLEAN_FEWSHOT_EXAMPLES[:num_fewshot])


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
    num_fewshot = max(1, NUM_FEWSHOT)
    if is_contam == 1 and CONTAM_MODE == "all_repeat":
        contam_examples = [{"question": question, "answer": answer} for _ in range(num_fewshot)]
        return examples_to_messages(contam_examples, is_contam)

    clean_examples = select_clean_fewshots(question, num_fewshot)
    if is_contam == 1:
        clean_examples = clean_examples[: max(0, num_fewshot - 1)]
        clean_examples.append({"question": question, "answer": answer})
    return examples_to_messages(clean_examples[:num_fewshot], is_contam)


def build_answer_messages(prefix_messages: Sequence[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    return list(prefix_messages) + [{"role": "user", "content": make_user_prompt(question)}]


def render_prompt(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def generate_completion(
    tokenizer,
    model,
    messages: Sequence[Dict[str, str]],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    prompt_text = render_prompt(tokenizer, messages)
    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = model_inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def score_response_from_messages(
    tokenizer,
    model,
    messages: Sequence[Dict[str, str]],
    response_text: str,
    temperature: float = 1.0,
) -> Dict:
    prefix_text = render_prompt(tokenizer, messages)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    full_text = prefix_text + response_text
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        shift_logits = outputs.logits[0, :-1, :].contiguous()
        shift_labels = full_ids[0, 1:].contiguous()
        target_logits = shift_logits[prefix_len - 1 :]
        target_labels = shift_labels[prefix_len - 1 :]
        if temperature != 1.0:
            target_logits = target_logits / temperature
        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)
        probs = torch.softmax(target_logits, dim=-1)
        token_mu = (probs * log_probs).sum(dim=-1)
        token_second_moment = (probs * torch.square(log_probs)).sum(dim=-1)
        token_sigma = torch.clamp(token_second_moment - torch.square(token_mu), min=0.0)

    token_log_probs_np = token_log_probs.detach().cpu().numpy()
    token_mu_np = token_mu.detach().cpu().numpy()
    token_sigma_np = token_sigma.detach().cpu().numpy()
    mean_logprob = float(token_log_probs_np.mean()) if len(token_log_probs_np) > 0 else float("-inf")
    sum_logprob = float(token_log_probs_np.sum()) if len(token_log_probs_np) > 0 else float("-inf")
    ppl = float(math.exp(-mean_logprob)) if np.isfinite(mean_logprob) else float("inf")
    return {
        "response_text": response_text,
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "ppl": ppl,
        "token_log_probs": token_log_probs_np.tolist(),
        "token_mu": token_mu_np.tolist(),
        "token_sigma": token_sigma_np.tolist(),
        "num_tokens": int(len(token_log_probs_np)),
        "prefix_text": prefix_text,
    }


def score_suffix_from_message_lists(
    tokenizer,
    model,
    prefix_messages: Sequence[Dict[str, str]],
    full_messages: Sequence[Dict[str, str]],
    temperature: float = 1.0,
) -> Dict:
    prefix_text = render_prompt(tokenizer, prefix_messages, add_generation_prompt=False)
    full_text = render_prompt(tokenizer, full_messages, add_generation_prompt=False)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        shift_logits = outputs.logits[0, :-1, :].contiguous()
        shift_labels = full_ids[0, 1:].contiguous()
        target_logits = shift_logits[prefix_len - 1 :]
        target_labels = shift_labels[prefix_len - 1 :]
        if temperature != 1.0:
            target_logits = target_logits / temperature
        log_probs = torch.log_softmax(target_logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1)

    token_log_probs_np = token_log_probs.detach().cpu().numpy()
    mean_logprob = float(token_log_probs_np.mean()) if len(token_log_probs_np) > 0 else float("-inf")
    sum_logprob = float(token_log_probs_np.sum()) if len(token_log_probs_np) > 0 else float("-inf")
    ppl = float(math.exp(-mean_logprob)) if np.isfinite(mean_logprob) else float("inf")
    return {
        "mean_logprob": mean_logprob,
        "sum_logprob": sum_logprob,
        "ppl": ppl,
        "token_log_probs": token_log_probs_np.tolist(),
        "num_tokens": int(len(token_log_probs_np)),
        "prefix_text": prefix_text,
        "full_text": full_text,
    }


def score_question_text_from_prefix(
    tokenizer,
    model,
    prefix_messages: Sequence[Dict[str, str]],
    question: str,
    temperature: float = 1.0,
) -> Dict:
    full_messages = list(prefix_messages) + [{"role": "user", "content": make_user_prompt(question)}]
    return score_suffix_from_message_lists(tokenizer, model, prefix_messages, full_messages, temperature)


def score_multiple_choice_letters(
    tokenizer,
    model,
    messages: Sequence[Dict[str, str]],
    letters: Sequence[str],
    temperature: float = 1.0,
) -> Dict[str, float]:
    scores = {}
    for letter in letters:
        result = score_response_from_messages(tokenizer, model, messages, letter, temperature)
        scores[letter] = result["sum_logprob"]
    return scores


def score_choices_from_messages(
    tokenizer,
    model,
    messages: Sequence[Dict[str, str]],
    choices: Sequence[str],
    temperature: float = 1.0,
) -> Dict[str, float]:
    scores = {}
    for choice in choices:
        result = score_response_from_messages(tokenizer, model, messages, choice, temperature)
        scores[choice] = result["sum_logprob"]
    return scores


def normalize_slot_guess_text(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    first_line = re.sub(r"^\(?[A-J]\)?[\.\):：]?\s*", "", first_line)
    return first_line.strip().lower()


def normalize_choice_logprobs(choice_scores: Dict[str, float]) -> Dict[str, float]:
    finite_items = {k: v for k, v in choice_scores.items() if np.isfinite(v)}
    if not finite_items:
        uniform = 1.0 / max(1, len(choice_scores))
        return {choice: uniform for choice in choice_scores}
    values = np.array([finite_items[k] for k in finite_items], dtype=float)
    max_val = float(values.max())
    probs = np.exp(values - max_val)
    probs = probs / probs.sum()
    normalized = {choice: 0.0 for choice in choice_scores}
    for choice, prob in zip(finite_items.keys(), probs):
        normalized[choice] = float(prob)
    return normalized


def normalize_choice_scores(letter_scores: Dict[str, float]) -> Dict[str, float]:
    values = np.array(list(letter_scores.values()), dtype=float)
    if not np.isfinite(values).any():
        uniform = 1.0 / max(1, len(letter_scores))
        return {letter: uniform for letter in letter_scores}
    finite_values = values[np.isfinite(values)]
    finite_min = float(finite_values.min())
    finite_max = float(finite_values.max())
    values = np.where(np.isposinf(values), finite_max, values)
    values = np.where(np.isneginf(values), finite_min, values)
    values = np.where(np.isnan(values), finite_min, values)
    max_val = float(values.max())
    probs = np.exp(values - max_val)
    probs = probs / probs.sum()
    return {letter: float(prob) for letter, prob in zip(letter_scores.keys(), probs)}


def build_metrics_from_scores(labels: Sequence[int], scores: Sequence[float]) -> Dict[str, Optional[float]]:
    labels_np = np.array(labels, dtype=int)
    scores_np = np.array(scores, dtype=float)
    finite_mask = np.isfinite(scores_np)
    nonfinite_count = int((~finite_mask).sum())
    if not finite_mask.all():
        print(f"[WARN] Replacing {nonfinite_count} non-finite score(s) before metric calculation.")
        if finite_mask.any():
            finite_scores = scores_np[finite_mask]
            finite_min = float(finite_scores.min())
            finite_max = float(finite_scores.max())
            scores_np = np.where(np.isposinf(scores_np), finite_max, scores_np)
            scores_np = np.where(np.isneginf(scores_np), finite_min, scores_np)
            scores_np = np.where(np.isnan(scores_np), finite_min, scores_np)
        else:
            scores_np = np.zeros_like(scores_np, dtype=float)
    unique_scores = np.unique(scores_np)
    best = None

    for threshold in unique_scores:
        preds = (scores_np >= threshold).astype(int)
        acc = float(accuracy_score(labels_np, preds))
        f1 = float(f1_score(labels_np, preds, zero_division=0))
        if best is None or f1 > best["F1 Score"] or (f1 == best["F1 Score"] and acc > best["Accuracy"]):
            best = {
                "threshold": float(threshold),
                "preds": preds.tolist(),
                "Accuracy": acc,
                "Precision": float(precision_score(labels_np, preds, zero_division=0)),
                "Recall": float(recall_score(labels_np, preds, zero_division=0)),
                "F1 Score": f1,
            }

    auc_value = None
    if len(np.unique(labels_np)) > 1:
        auc_value = float(roc_auc_score(labels_np, scores_np))

    return {
        "Accuracy": best["Accuracy"],
        "Precision": best["Precision"],
        "Recall": best["Recall"],
        "F1 Score": best["F1 Score"],
        "AUC": auc_value,
        "Best Threshold": best["threshold"],
        "Pred Labels": best["preds"],
        "Non-finite Scores": nonfinite_count,
    }


def get_rows(json_path: str, limit: Optional[int] = None) -> List[Dict]:
    rows = load_jsonl(json_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def build_rewrite_lookup_key(question: str, num_variants: int = 4) -> str:
    task_profile = infer_task_profile(question)
    return f"{task_profile}::n={num_variants}::{question}"


def get_generated_variants(question: str, num_variants: int) -> Optional[List[str]]:
    key = build_rewrite_lookup_key(question, num_variants=num_variants)
    payload = GENERATED_REWRITES.get(key)
    if not isinstance(payload, dict):
        return None
    variants = payload.get("variants")
    if isinstance(variants, list):
        filtered = filter_valid_rewrite_variants(question, variants, num_variants)
        if len(filtered) >= num_variants:
            return filtered[:num_variants]
    return None


def record_generated_variants(question: str, answer: str, variants: Sequence[str], source: str, num_variants: int) -> None:
    filtered_variants = filter_valid_rewrite_variants(question, variants, num_variants=max(num_variants, len(variants)))
    key = build_rewrite_lookup_key(question, num_variants=num_variants)
    GENERATED_REWRITES[key] = {
        "question": question,
        "answer": answer,
        "task_profile": infer_task_profile(question),
        "variants": [str(item) for item in filtered_variants],
        "source": source,
    }


def save_generated_rewrites(path: Optional[str], json_path: str, num_rows: int, harness_task: str, num_variants: int) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "json_path": json_path,
            "num_rows": num_rows,
            "num_unique_questions": len(GENERATED_REWRITES),
            "num_variants": num_variants,
            "harness_task": harness_task,
        },
        "items": GENERATED_REWRITES,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_print_first_prompt(
    method_name: str,
    tokenizer,
    prefix_messages: Sequence[Dict[str, str]],
    full_messages: Sequence[Dict[str, str]],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not PRINT_FIRST_PROMPTS:
        return
    print(f"\n===== FIRST SAMPLE: {method_name} =====")
    print("\n--- FEWSHOT PREFIX ---\n")
    print(render_prompt(tokenizer, prefix_messages, add_generation_prompt=False))
    print("\n--- FULL PROMPT ---\n")
    print(render_prompt(tokenizer, full_messages))
    if extra:
        print("\n--- EXTRA ---\n")
        print(json.dumps(extra, ensure_ascii=False, indent=2))


def split_mmlu_input(text: str) -> Tuple[str, Dict[str, str]]:
    lines = text.strip().splitlines()
    stem_lines = []
    options = {}
    for line in lines:
        stripped = line.strip()
        if stripped in {"A:", "Answer:"}:
            stem_lines.append(stripped)
            continue
        match = re.match(r"^(?:\(([A-J])\)|([A-J])[\.\):?])\s*(.*)$", stripped)
        if match:
            letter = match.group(1) or match.group(2)
            options[letter] = match.group(3).strip()
        else:
            stem_lines.append(stripped)
    stem = "\n".join([line for line in stem_lines if line])
    return stem, options


def rebuild_mmlu_input(stem: str, options: Dict[str, str]) -> str:
    raw_lines = str(stem).splitlines() if stem else []
    lines = [line.rstrip() for line in raw_lines if line.strip()]
    answer_cue = ""
    if lines and lines[-1].strip() in {"A:", "Answer:"}:
        answer_cue = lines.pop(-1).strip()
    task_profile = infer_task_profile(stem)
    option_template = "{letter}. {text}" if task_profile == "mmlu_pro" else "({letter}) {text}"
    for letter in sorted(options.keys()):
        lines.append(option_template.format(letter=letter, text=options[letter]))
    if answer_cue:
        lines.append(answer_cue)
    return "\n".join(lines)


def extract_options_from_row(row: Dict) -> Dict[str, str]:
    if "options" in row and isinstance(row["options"], (list, tuple)):
        return {chr(ord("A") + i): str(value).strip() for i, value in enumerate(row["options"])}
    if "choices" in row and isinstance(row["choices"], (list, tuple)):
        return {chr(ord("A") + i): str(value).strip() for i, value in enumerate(row["choices"])}
    option_map = {}
    for letter in "ABCDEFGHIJ":
        if letter in row:
            option_map[letter] = str(row[letter]).strip()
    return option_map


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
    if explicit_question:
        return explicit_question
    if HARNESS_TASK_CONFIG:
        if harness_text:
            return harness_text
    input_text = str(row.get("input", "")).strip()
    stem_from_input, options_from_input = split_mmlu_input(input_text) if input_text else ("", {})
    if options_from_input:
        return input_text
    options = extract_options_from_row(row)
    stem = explicit_question or stem_from_input or input_text
    if options:
        return rebuild_mmlu_input(stem, options)
    return stem


def build_answer_from_row(row: Dict) -> str:
    explicit_answer = str(row.get("answer", "")).strip()
    if explicit_answer:
        return explicit_answer
    if HARNESS_TASK_CONFIG:
        answer = harness_doc_to_target(row)
        if answer:
            return str(answer).strip()
    return str(row.get("target", "")).strip()


def normalize_answer_label(answer: str) -> str:
    match = re.match(r"^\(?([A-J])\)?$", str(answer).strip())
    if match:
        return match.group(1)
    return str(answer).strip()


def build_full_answer_text(question: str, answer: str, row: Optional[Dict] = None) -> str:
    task_profile = infer_task_profile(question)
    if task_profile in {"bbh_date_understanding", "mmlu_pro"}:
        return str(answer).strip()
    if task_profile == "bbh_object_counting":
        return str(answer).strip()
    _, options = split_mmlu_input(question)
    if not options and row is not None:
        options = extract_options_from_row(row)
    answer_label = normalize_answer_label(answer)
    if answer_label in options:
        return f"({answer_label}) {options[answer_label]}"
    return str(answer).strip()


def choose_mask_span(question: str, answer: str, row: Optional[Dict] = None) -> Tuple[str, str]:
    stem, options = split_mmlu_input(question)
    if not options and row is not None:
        options = extract_options_from_row(row)
        if not stem:
            stem = str(row.get("question", "")).strip()
    if not options:
        fallback_text = stem or question
        tokens = fallback_text.split()
        if not tokens:
            doc_id = row.get("doc_id", "unknown") if row is not None else "unknown"
            raise ValueError(f"Failed to parse answer options and stem for date sample doc_id={doc_id}")

        candidate_indices = [
            i for i, tok in enumerate(tokens)
            if len(tok.strip(".,:;!?()[]{}\"'")) >= 5 and not tok.endswith("?")
        ]
        target_idx = candidate_indices[len(candidate_indices) // 2] if candidate_indices else max(0, len(tokens) // 2)
        gold_span = tokens[target_idx]
        masked_tokens = list(tokens)
        masked_tokens[target_idx] = "[MASK]"
        return " ".join(masked_tokens), gold_span

    normalized_answer = str(answer).strip().strip("()")
    wrong_letters = [letter for letter in sorted(options.keys()) if letter != normalized_answer]
    target_letter = wrong_letters[0] if wrong_letters else sorted(options.keys())[0]
    gold_span = options[target_letter]
    masked_lines = [stem] if stem else []
    for letter in sorted(options.keys()):
        option_text = "[MASK]" if letter == target_letter else options[letter]
        masked_lines.append(f"({letter}) {option_text}")
    return "\n".join(masked_lines), gold_span


def perturb_stem(stem: str, variant_index: int, task_profile: str = "generic") -> str:
    if task_profile == "bbh_date_understanding":
        replacements_list = [
            {
                "What is the date": "Which date is",
                "in MM/DD/YYYY": "using MM/DD/YYYY",
                "Today is": "Assume today is",
                "Tomorrow is": "Assume tomorrow is",
                "yesterday": "the previous day",
            },
            {
                "What is the date": "Determine the date",
                "tomorrow": "the next day",
                "10 days ago": "ten days before today",
                "one year ago from today": "one year before today",
                "MM/DD/YYYY": "month/day/year format",
            },
            {
                "What is the date": "Select the date",
                "Jane is correct.": "Jane's date is correct.",
                "John is correct.": "John's date is correct.",
                "It is their": "Today is their",
                "anniversary today": "anniversary",
            },
            {
                "What is the date": "Identify the date",
                "Today is": "The current day is",
                "Tomorrow is": "The following day is",
                "in MM/DD/YYYY": "formatted as MM/DD/YYYY",
                "from today": "relative to today",
            },
        ]
    elif task_profile == "bbh_object_counting":
        replacements_list = [
            {
                "How many": "Determine how many",
                "do I have": "are there",
            },
            {
                "How many": "Count how many",
                "I have": "are listed",
            },
            {
                "How many": "Identify how many",
                "do I have": "I am listing",
            },
            {
                "How many": "What number of",
                "do I have": "are present",
            },
        ]
    else:
        replacements_list = [
            {
                "Which of the following": "Which option",
                "What is": "Determine",
                "Find": "Determine",
                "Calculate": "Compute",
                "Pick the correct": "Select the correct",
            },
            {
                "Which statement": "Which choice",
                "correctly": "accurately",
                "following": "listed",
                "Determine": "Find",
                "Compute": "Calculate",
            },
            {
                "What percent": "Which percent",
                "Find the": "Determine the",
                "What will": "Determine",
                "Compare the": "Evaluate the",
                "Choose": "Select",
            },
            {
                "What is the": "Identify the",
                "What does": "Identify what",
                "How much": "Determine how much",
                "How many": "Determine how many",
                "What were": "Determine",
            },
        ]
    updated = stem
    replacements = replacements_list[variant_index % len(replacements_list)]
    replaced = False
    for src, dst in replacements.items():
        if src in updated:
            updated = updated.replace(src, dst, 1)
            replaced = True
            break
    if not replaced:
        if task_profile == "bbh_date_understanding":
            prefixes = [
                "Infer the answer for this date question: ",
                "Read the context and infer the date: ",
                "Use the temporal context to answer: ",
                "Solve this date-understanding question: ",
            ]
        elif task_profile == "bbh_object_counting":
            prefixes = [
                "Count the requested objects in this question: ",
                "Read the list carefully and answer: ",
                "Solve this counting question: ",
                "Determine the requested count: ",
            ]
        else:
            prefixes = [
                "Consider the following problem: ",
                "Answer the following question: ",
                "Select the best response for: ",
                "Review this problem and answer it: ",
            ]
        updated = prefixes[variant_index % len(prefixes)] + updated
    return updated


def generate_dcq_variants(question: str, answer: str, seed: int, num_variants: int = 4) -> List[str]:
    generated = get_generated_variants(question, num_variants)
    if generated is not None:
        return generated
    variants = generate_heuristic_dcq_variants(question, seed, num_variants=num_variants)
    record_generated_variants(question, answer, variants, "heuristic", num_variants)
    return variants


def infer_task_profile(text: str = "") -> str:
    task_name = ""
    if HARNESS_TASK_CONFIG:
        task_name = str(HARNESS_TASK_CONFIG.get("task", "")).lower()
    system_prompt = PROMPT_SYSTEM_CONTENT.strip().lower()
    if "object_counting" in task_name or "counting" in task_name or "enumerating objects" in system_prompt:
        return "bbh_object_counting"
    if "date_understanding" in task_name or "infer the date from context" in system_prompt:
        return "bbh_date_understanding"
    if "mmlu_pro" in task_name:
        return "mmlu_pro"
    _, options = split_mmlu_input(text)
    if options:
        return "multiple_choice"
    return "generic"


def describe_detection_instance(text: str) -> str:
    profile = infer_task_profile(text)
    if profile == "bbh_object_counting":
        return "object-counting question"
    if profile in {"bbh_date_understanding", "mmlu_pro", "multiple_choice"}:
        return "multiple-choice question"
    return "question"


def build_rewrite_system_prompt(task_profile: str) -> str:
    del task_profile
    return (
        "You rewrite evaluation questions into several natural alternative phrasings. "
        "Keep the task and answerability intact, but actually rewrite the wording. "
        "Do not explain your work. Output valid JSON only."
    )


def build_rewrite_task_notes(task_profile: str) -> str:
    if task_profile == "bbh_date_understanding":
        return (
            "Task-specific constraints: preserve the date-understanding meaning, temporal relations, date values, answer-choice labels, "
            "and the number of answer choices. You may rewrite both the question text and the answer-choice text."
        )
    if task_profile == "bbh_object_counting":
        return (
            "Task-specific constraints: preserve the counting meaning, all listed objects, quantities, and the number of answer choices if there are any. "
            "You may rewrite the full question text."
        )
    if task_profile in {"mmlu_pro", "multiple_choice"}:
        return (
            "Task-specific constraints: preserve the question meaning, answer-choice labels, and the number of answer choices. "
            "You may rewrite both the question text and the answer-choice text."
        )
    return (
        "Task-specific constraints: preserve all task-critical entities, numbers, labels, and answer-defining content exactly."
    )


def build_rewrite_user_prompt(question: str, num_variants: int, task_profile: str) -> str:
    task_notes = build_rewrite_task_notes(task_profile)
    return (
        "Rewrite the following evaluation question into {n} different natural variants.\n\n"
        "Requirements:\n"
        "1. Keep the original meaning, task, and answerability unchanged.\n"
        "2. Keep the same answer-choice labels and the same number of answer choices, if the question has choices.\n"
        "3. You may rewrite both the question text and the answer-choice text.\n"
        "4. Each rewrite must be clearly different from the original and from the other rewrites.\n"
        "5. Do not leave any answer choice blank.\n"
        "6. Do not add explanations, notes, or surrounding commentary.\n"
        "7. Return full rewritten questions only.\n"
        "{task_notes}\n\n"
        "Return JSON only in this format:\n"
        "{{\"variants\": [\"rewrite 1\", \"rewrite 2\", \"rewrite 3\"]}}\n\n"
        "Bad outputs:\n"
        "- copying the original question unchanged\n"
        "- returning duplicate rewrites\n"
        "- deleting answer choices or labels\n\n"
        "Original question:\n{question}\n"
    ).format(n=num_variants, question=question, task_notes=task_notes)


def build_dcq_dataset_descriptor() -> Tuple[str, str]:
    task_name = (HARNESS_TASK_NAME or "").strip()
    if not task_name and HARNESS_TASK_CONFIG:
        task_name = str(HARNESS_TASK_CONFIG.get("_task_name") or "").strip()
    if task_name.startswith("leaderboard_"):
        task_name = task_name[len("leaderboard_") :]
    if task_name:
        return task_name, "evaluation"
    dataset_name = Path(JSON_PATH).stem if JSON_PATH else "evaluation dataset"
    return dataset_name, "evaluation"


def extract_answer_cue(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        if line in {"A:", "Answer:"}:
            return line
    return ""


def extract_prefix_marker(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if first.startswith("Q:"):
        return "Q:"
    if first.startswith("Problem:"):
        return "Problem:"
    return ""


def is_valid_rewrite_variant(original_question: str, variant: str, task_profile: str) -> bool:
    original_question = str(original_question).strip()
    variant = str(variant).strip()
    if not variant or variant == original_question:
        return False

    orig_stem, orig_options = split_mmlu_input(original_question)
    var_stem, var_options = split_mmlu_input(variant)

    if orig_options:
        if not var_options:
            return False
        if list(sorted(orig_options.keys())) != list(sorted(var_options.keys())):
            return False
        if len(var_options) != len(orig_options):
            return False
        if any(not str(text).strip() for text in var_options.values()):
            return False
        if not (var_stem or "").strip():
            return False
        return True

    if task_profile == "bbh_object_counting":
        return bool((var_stem or variant).strip())

    return bool(variant)


def filter_valid_rewrite_variants(original_question: str, variants: Sequence[str], num_variants: int) -> List[str]:
    task_profile = infer_task_profile(original_question)
    valid: List[str] = []
    seen = {str(original_question).strip()}
    for item in variants:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if not is_valid_rewrite_variant(original_question, text, task_profile):
            continue
        valid.append(text)
        if len(valid) >= num_variants:
            break
    return valid


def build_dcq_quiz_prompt_lines(options: List[str]) -> List[str]:
    dataset_name, split_name = build_dcq_dataset_descriptor()
    return [
        "You are provided with a five-choice quiz. "
        f"Your task is to CORRECTLY select the option that EXACTLY corresponds to an instance from the {split_name} split of the {dataset_name} dataset.",
        "",
        "When selecting the option, you MUST ensure that you follow the following rules:",
        "(1) You MUST ensure that you ONLY generate a single option letter as your answer.",
        '(2) If you do not know the dataset or the correct answer, you MUST select option "E) None of the provided options."',
        "",
        "Hint: While all the following options seem similar in terms of meaning, there is ONLY one option that reflects the EXACT WORDING with respect to the original instance.",
        "---",
        "A)",
        options[0],
        "",
        "B)",
        options[1],
        "",
        "C)",
        options[2],
        "",
        "D)",
        options[3],
        "",
        "E)",
        "None of the provided options.",
        "---",
        "ANSWER:",
    ]


def generate_heuristic_dcq_variants(question: str, seed: int, num_variants: int = 4) -> List[str]:
    task_profile = infer_task_profile(question)
    stem, options = split_mmlu_input(question)
    candidates = []
    seen = {question}
    for idx in range(max(num_variants * 4, 16)):
        variant_stem = perturb_stem(stem, idx, task_profile=task_profile)
        variant = rebuild_mmlu_input(variant_stem, options)
        if variant not in seen:
            candidates.append(variant)
            seen.add(variant)

    if task_profile == "bbh_date_understanding":
        fallback_stems = [
            "Infer the date from context.\n" + stem,
            "Select the correct date for the following question.\n" + stem,
            "Read the temporal context carefully and choose the correct answer.\n" + stem,
            "Choose the correct date for this problem.\n" + stem,
            stem + "\nPlease select the best answer choice.",
            stem + "\nPick the single best option.",
            "Question:\n" + stem + "\nChoose one date option.",
            "Consider the following date-understanding question.\n" + stem,
        ]
    elif task_profile == "bbh_object_counting":
        fallback_stems = [
            "Count the relevant objects in the following question.\n" + stem,
            "Read the list carefully and determine the correct count.\n" + stem,
            "Choose the correct count for this problem.\n" + stem,
            stem + "\nPlease provide the correct count.",
        ]
    else:
        fallback_stems = [
            "Answer the following multiple-choice question:\n" + stem,
            "Select the best option for the following question.\n" + stem,
            "Read the question carefully and choose the correct answer.\n" + stem,
            "Choose the correct answer for this problem.\n" + stem,
            stem + "\nPlease select the best answer choice.",
            stem + "\nPick the single best option.",
            "Question:\n" + stem + "\nChoose one option.",
            "Consider the following exam-style question.\n" + stem,
        ]
    for fallback_stem in fallback_stems:
        variant = rebuild_mmlu_input(fallback_stem, options)
        if variant not in seen:
            candidates.append(variant)
            seen.add(variant)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if len(candidates) < num_variants:
        while len(candidates) < num_variants:
            suffix = f"\nVariant instruction {len(candidates) + 1}: choose the correct answer."
            variant = rebuild_mmlu_input(stem + suffix, options)
            if variant not in seen:
                candidates.append(variant)
                seen.add(variant)
    return candidates[:num_variants]


def build_ts_guessing_messages(prefix_messages, masked_question: str):
    task_profile = infer_task_profile(masked_question)
    if task_profile in {"bbh_date_understanding", "mmlu_pro", "multiple_choice"}:
        prompt = (
            "One wrong answer option in the multiple-choice question below has been replaced with [MASK]. "
            "Do not answer the question. Instead, guess the exact missing wrong option text. "
            "Return only the missing option text.\n"
            f"{masked_question}\nAnswer:"
        )
    else:
        instance_type = describe_detection_instance(masked_question)
        prompt = (
            f"One span in the {instance_type} below has been replaced with [MASK]. "
            "Do not answer the question. Instead, recover the exact missing span from the original question text. "
            "Return only the missing span.\n"
            f"{masked_question}\nAnswer:"
        )
    return list(prefix_messages) + [{"role": "user", "content": prompt}]


def build_decop_messages(prefix_messages, options: List[str]):
    task_profile = infer_task_profile(options[0])
    if task_profile == "bbh_date_understanding":
        prompt_lines = [
            "Choose the option that exactly matches a dataset instance that appeared in the examples above.",
            "Return only one letter: A, B, C, or D.",
            "",
            f"A. {options[0]}",
            f"B. {options[1]}",
            f"C. {options[2]}",
            f"D. {options[3]}",
            "",
            "Answer:",
        ]
    elif task_profile == "mmlu_pro":
        prompt_lines = [
            "Choose the option that exactly matches a dataset instance that appeared in the examples above.",
            "Return only one letter: A, B, C, or D.",
            "",
            f"A. {options[0]}",
            f"B. {options[1]}",
            f"C. {options[2]}",
            f"D. {options[3]}",
            "",
            "Answer:",
        ]
    else:
        instance_type = describe_detection_instance(options[0])
        prompt_lines = [
            f"Each option below is a complete {instance_type}.",
            "Choose the option whose full wording exactly matches an example shown above.",
            f"Do not answer any {instance_type}. Compare the full question text only.",
            "Return only one letter: A, B, C, or D.",
            "",
            f"A. {options[0]}",
            f"B. {options[1]}",
            f"C. {options[2]}",
            f"D. {options[3]}",
            "",
            "Answer:",
        ]
    return list(prefix_messages) + [{"role": "user", "content": "\n".join(prompt_lines)}]


def build_dcq_messages(prefix_messages, options: List[str]):
    prompt_lines = build_dcq_quiz_prompt_lines(options)
    return list(prefix_messages) + [{"role": "user", "content": "\n".join(prompt_lines)}]


def build_dcq_bias_messages(prefix_messages, options: List[str]):
    prompt_lines = build_dcq_quiz_prompt_lines(options)
    return list(prefix_messages) + [{"role": "user", "content": "\n".join(prompt_lines)}]


def finalize_method_results(
    method_name: str,
    labels: Sequence[int],
    scores: Sequence[float],
    rows: List[Dict],
    summary_extra: Optional[Dict] = None,
) -> Dict:
    metrics = build_metrics_from_scores(labels, scores)
    pred_labels = metrics.pop("Pred Labels")
    scores_np = np.array(scores, dtype=float)
    finite_scores = scores_np[np.isfinite(scores_np)]
    for row, pred in zip(rows, pred_labels):
        row["pred_label"] = pred
    summary = {
        "method": method_name,
        "num_rows": len(rows),
        "avg_score": float(finite_scores.mean()) if len(finite_scores) else None,
        "num_nonfinite_scores": int((~np.isfinite(scores_np)).sum()),
        "metrics": metrics,
    }
    if summary_extra:
        summary.update(summary_extra)
    return {"rows": rows, "summary": summary}


def run_perplexity_baseline(tokenizer, model, rows: Sequence[Dict], temperature: float) -> Dict:
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="Perplexity", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        full_answer = build_full_answer_text(question, answer, row)
        label = normalize_label(row["is_contam"])
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_answer_messages(prefix_messages, question)
        if idx == 0:
            maybe_print_first_prompt("Perplexity", tokenizer, prefix_messages, messages, {"scored_text": full_answer})
        answer_score = score_response_from_messages(tokenizer, model, messages, full_answer, temperature)
        score = answer_score["mean_logprob"]
        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "full_answer_text": full_answer,
                "gold_label": label,
                "score": score,
                "perplexity": answer_score["ppl"],
                "token_log_probs": answer_score["token_log_probs"],
                "num_tokens": answer_score["num_tokens"],
                "scored_text": full_answer,
            }
        )
    return finalize_method_results("Perplexity", labels, scores, results, {"temperature": temperature})


def run_min_k_baseline(tokenizer, model, rows: Sequence[Dict], temperature: float, k_ratio: float) -> Dict:
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="Min-k", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        full_answer = build_full_answer_text(question, answer, row)
        label = normalize_label(row["is_contam"])
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_answer_messages(prefix_messages, question)
        if idx == 0:
            maybe_print_first_prompt("Min-k% Prob", tokenizer, prefix_messages, messages, {"scored_text": full_answer, "k_ratio": k_ratio})
        answer_score = score_response_from_messages(tokenizer, model, messages, full_answer, temperature)
        token_log_probs = answer_score["token_log_probs"]
        k = max(1, int(len(token_log_probs) * k_ratio))
        score = float(sum(sorted(token_log_probs)[:k]) / k)
        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "full_answer_text": full_answer,
                "gold_label": label,
                "score": score,
                "k_ratio": k_ratio,
                "k": k,
                "token_log_probs": token_log_probs,
                "num_tokens": answer_score["num_tokens"],
                "scored_text": full_answer,
            }
        )
    return finalize_method_results("Min-k% Prob", labels, scores, results, {"temperature": temperature, "k_ratio": k_ratio})


def run_min_k_plus_plus_baseline(
    tokenizer,
    model,
    rows: Sequence[Dict],
    temperature: float,
    k_ratio: float,
) -> Dict:
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="Min-k++", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        full_answer = build_full_answer_text(question, answer, row)
        label = normalize_label(row["is_contam"])
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_answer_messages(prefix_messages, question)
        if idx == 0:
            maybe_print_first_prompt("Min-K%++", tokenizer, prefix_messages, messages, {"scored_text": full_answer, "k_ratio": k_ratio})
        answer_score = score_response_from_messages(tokenizer, model, messages, full_answer, temperature)
        token_log_probs = np.array(answer_score["token_log_probs"], dtype=float)
        token_mu = np.array(answer_score["token_mu"], dtype=float)
        token_sigma = np.array(answer_score["token_sigma"], dtype=float)
        normalized_token_scores = (token_log_probs - token_mu) / np.sqrt(np.maximum(token_sigma, 1e-12))
        k = max(1, int(len(normalized_token_scores) * k_ratio))
        score = float(np.sort(normalized_token_scores)[:k].mean())
        labels.append(label)
        scores.append(score)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "full_answer_text": full_answer,
                "gold_label": label,
                "score": score,
                "k_ratio": k_ratio,
                "k": k,
                "token_log_probs": token_log_probs.tolist(),
                "token_mu": token_mu.tolist(),
                "token_sigma": token_sigma.tolist(),
                "token_mink_plus_plus": normalized_token_scores.tolist(),
                "num_tokens": answer_score["num_tokens"],
                "scored_text": full_answer,
            }
        )
    return finalize_method_results(
        "Min-K%++",
        labels,
        scores,
        results,
        {"temperature": temperature, "k_ratio": k_ratio},
    )


def run_ts_guessing_baseline(
    tokenizer,
    model,
    rows: Sequence[Dict],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Dict:
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="TS-Guessing", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        label = normalize_label(row["is_contam"])
        prefix_messages = build_prefix_messages(question, answer, label)
        masked_question, masked_span = choose_mask_span(question, answer, row)
        messages = build_ts_guessing_messages(prefix_messages, masked_question)
        if idx == 0:
            maybe_print_first_prompt("TS-Guessing", tokenizer, prefix_messages, messages, {"gold_span": masked_span})
        greedy_guess = generate_completion(
            tokenizer,
            model,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            top_p=top_p,
        )
        score = float(normalize_slot_guess_text(greedy_guess) == normalize_slot_guess_text(masked_span))
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
                "mode": "paper_ts_guessing",
                "token_log_probs": None,
                "num_tokens": None,
            }
        )
    return finalize_method_results(
        "TS-Guessing",
        labels,
        scores,
        results,
        {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens},
    )


def run_dcq_decop_baseline(tokenizer, model, rows: Sequence[Dict], temperature: float, seed: int) -> Dict:
    option_letters = ["A", "B", "C", "D", "E"]
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="DCQ/DE-COP", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        label = normalize_label(row["is_contam"])
        variants = generate_dcq_variants(question, answer, seed=seed + idx, num_variants=3)
        options = [question] + variants
        rng_local = random.Random(seed + idx)
        rng_local.shuffle(options)
        correct_letter = option_letters[options.index(question)]
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_dcq_messages(prefix_messages, options)
        letter_scores = score_multiple_choice_letters(tokenizer, model, messages, option_letters, temperature)
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
                "options": {letter: option for letter, option in zip(option_letters[:4], options)},
                "correct_letter": correct_letter,
                "chosen_letter": chosen_letter,
                "score": score,
                "letter_scores": letter_scores,
                "letter_probs": letter_probs,
            }
        )
    return finalize_method_results("DCQ / DE-COP", labels, scores, results, {"temperature": temperature})


def run_decop_baseline(tokenizer, model, rows: Sequence[Dict], temperature: float, seed: int) -> Dict:
    option_letters = ["A", "B", "C", "D"]
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="DE-COP", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        label = normalize_label(row["is_contam"])
        variants = generate_dcq_variants(question, answer, seed=seed + idx, num_variants=3)
        options = [question] + variants
        rng_local = random.Random(seed + idx)
        rng_local.shuffle(options)
        correct_letter = option_letters[options.index(question)]
        prefix_messages = build_prefix_messages(question, answer, label)
        messages = build_decop_messages(prefix_messages, options)
        if idx == 0:
            maybe_print_first_prompt("DE-COP", tokenizer, prefix_messages, messages, {"options": options})
        letter_scores = score_multiple_choice_letters(tokenizer, model, messages, option_letters, temperature)
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
                "options": {letter: option for letter, option in zip(option_letters, options)},
                "correct_letter": correct_letter,
                "chosen_letter": chosen_letter,
                "score": score,
                "letter_scores": letter_scores,
                "letter_probs": letter_probs,
            }
        )
    return finalize_method_results("DE-COP", labels, scores, results, {"temperature": temperature})


def run_dcq_baseline(tokenizer, model, rows: Sequence[Dict], temperature: float, seed: int) -> Dict:
    option_letters = ["A", "B", "C", "D", "E"]
    labels, scores, results = [], [], []
    for idx, row in enumerate(tqdm(rows, desc="DCQ", leave=False)):
        question = build_question_from_row(row)
        answer = build_answer_from_row(row)
        label = normalize_label(row["is_contam"])
        variants = generate_dcq_variants(question, answer, seed=seed + idx)
        prefix_messages = build_prefix_messages(question, answer, label)

        # Bias detector quiz: all A-D are perturbations.
        bdq_messages = build_dcq_bias_messages(prefix_messages, variants)
        if idx == 0:
            maybe_print_first_prompt("DCQ-BDQ", tokenizer, prefix_messages, bdq_messages, {"variants": variants})
        bdq_letter_scores = score_multiple_choice_letters(tokenizer, model, bdq_messages, option_letters, temperature)
        bdq_letter_probs = normalize_choice_scores(bdq_letter_scores)

        random_chance = 1.0 / len(option_letters)
        non_preferred_letters = [letter for letter in option_letters[:4] if bdq_letter_probs.get(letter, 0.0) < random_chance]
        if not non_preferred_letters:
            ranked_letters = sorted(option_letters[:4], key=lambda l: bdq_letter_probs[l])
            non_preferred_letters = ranked_letters[:1]
        bcq_runs = []
        best_prob = 0.0
        best_run = None
        bdq_options_map = {letter: variant for letter, variant in zip(option_letters[:4], variants)}

        for target_letter in non_preferred_letters:
            options_map = dict(bdq_options_map)
            options_map[target_letter] = question
            options = [options_map["A"], options_map["B"], options_map["C"], options_map["D"]]
            messages = build_dcq_messages(prefix_messages, options)
            if idx == 0 and not bcq_runs:
                maybe_print_first_prompt("DCQ-BCQ", tokenizer, prefix_messages, messages, {"target_letter": target_letter, "options": options_map})
            letter_scores = score_multiple_choice_letters(tokenizer, model, messages, option_letters, temperature)
            letter_probs = normalize_choice_scores(letter_scores)
            prob = letter_probs[target_letter]
            run_info = {
                "target_letter": target_letter,
                "options": options_map,
                "letter_scores": letter_scores,
                "letter_probs": letter_probs,
                "score": prob,
            }
            bcq_runs.append(run_info)
            if prob > best_prob:
                best_prob = prob
                best_run = run_info

        labels.append(label)
        scores.append(best_prob)
        results.append(
            {
                "doc_id": row.get("doc_id", idx),
                "question": question,
                "answer": answer,
                "gold_label": label,
                "bdq_letter_scores": bdq_letter_scores,
                "bdq_letter_probs": bdq_letter_probs,
                "non_preferred_letters": non_preferred_letters,
                "bcq_runs": bcq_runs,
                "score": best_prob,
                "best_target_letter": best_run["target_letter"] if best_run else None,
            }
        )
    return finalize_method_results("DCQ", labels, scores, results, {"temperature": temperature})


def print_results_table(method_summaries: Sequence[Dict]) -> None:
    rows = []
    for summary in method_summaries:
        metrics = summary["metrics"]
        auc_str = "None" if metrics["AUC"] is None else f"{metrics['AUC']:.4f}"
        rows.append(
            [
                summary["method"],
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['F1 Score']:.4f}",
                auc_str,
            ]
        )

    method_width = max(len("Method"), max(len(r[0]) for r in rows))
    col_width = max(len("Accuracy"), len("F1 Score"), len("AUC"), 8)
    header = f"{'Method'.ljust(method_width)}  {'Accuracy'.rjust(col_width)}  {'F1 Score'.rjust(col_width)}  {'AUC'.rjust(col_width)}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for method, acc, f1, auc in rows:
        print(f"{method.ljust(method_width)}  {acc.rjust(col_width)}  {f1.rjust(col_width)}  {auc.rjust(col_width)}")
    print(sep)


def main():
    global NUM_FEWSHOT, CONTAM_MODE, PROMPT_MODE
    global SAVE_REWRITES_PATH, GENERATED_REWRITES, PRINT_FIRST_PROMPTS
    parser = argparse.ArgumentParser(description="Run all baseline methods and print a unified result table.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--k_ratio", type=float, default=K_RATIO)
    parser.add_argument("--num_fewshot", type=int, default=NUM_FEWSHOT)
    parser.add_argument("--contam_mode", type=str, default=CONTAM_MODE, choices=["single_repeat", "all_repeat"])
    parser.add_argument("--prompt_mode", type=str, default=PROMPT_MODE, choices=["auto", "fixed"])
    parser.add_argument("--harness_dir", type=str, default=HARNESS_DIR)
    parser.add_argument("--harness_task", type=str, default=HARNESS_TASK)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--save_rewrites_path", type=str, default=None)
    parser.add_argument("--print_first_prompts", action="store_true")
    args = parser.parse_args()

    set_random_seed(args.seed)
    NUM_FEWSHOT = args.num_fewshot
    CONTAM_MODE = args.contam_mode
    PROMPT_MODE = args.prompt_mode
    SAVE_REWRITES_PATH = args.save_rewrites_path
    PRINT_FIRST_PROMPTS = args.print_first_prompts
    GENERATED_REWRITES = {}
    rows = get_rows(args.json_path, args.limit)
    configure_prompt_template(rows, args.json_path, args.num_fewshot, args.prompt_mode, args.harness_dir, args.harness_task)
    if HARNESS_TASK_CONFIG is None and len(CLEAN_FEWSHOT_EXAMPLES) < args.num_fewshot:
        print(
            f"[WARN] Only {len(CLEAN_FEWSHOT_EXAMPLES)} clean few-shot examples were prepared, "
            f"but --num_fewshot={args.num_fewshot}. Generic fallback examples may be used."
        )
    tokenizer, model = load_model_and_tokenizer(args.model_path)

    results_dir = args.results_dir or str(Path("mmlu_baseline_test") / get_dataset_model_tag(args.json_path, args.model_path))

    method_outputs = []
    method_outputs.append(run_perplexity_baseline(tokenizer, model, rows, args.temperature))
    method_outputs.append(run_min_k_baseline(tokenizer, model, rows, args.temperature, args.k_ratio))
    method_outputs.append(run_min_k_plus_plus_baseline(tokenizer, model, rows, args.temperature, args.k_ratio))
    method_outputs.append(run_ts_guessing_baseline(tokenizer, model, rows, args.temperature, args.top_p, args.max_new_tokens))
    method_outputs.append(run_decop_baseline(tokenizer, model, rows, args.temperature, args.seed))
    method_outputs.append(run_dcq_baseline(tokenizer, model, rows, args.temperature, args.seed))
    save_generated_rewrites(args.save_rewrites_path, args.json_path, len(rows), args.harness_task, 4)

    method_summaries = []
    for output in method_outputs:
        method_name = output["summary"]["method"]
        file_stem = method_name.lower().replace(" ", "_").replace("/", "_").replace("%", "pct").replace("-", "_")
        rows_path = str(Path(results_dir) / f"{file_stem}_results.jsonl")
        summary_path = str(Path(results_dir) / f"{file_stem}_summary.json")
        save_jsonl(rows_path, output["rows"])
        save_json(summary_path, output["summary"])
        method_summaries.append(output["summary"])

    combined_summary = {
        "model_path": args.model_path,
        "json_path": args.json_path,
        "num_rows": len(rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "k_ratio": args.k_ratio,
        "num_fewshot": args.num_fewshot,
        "contam_mode": args.contam_mode,
        "prompt_mode": args.prompt_mode,
        "prompt_system": PROMPT_SYSTEM_CONTENT,
        "harness_dir": args.harness_dir,
        "harness_task": args.harness_task,
        "harness_task_config": HARNESS_TASK_CONFIG.get("task") if HARNESS_TASK_CONFIG else None,
        "fallback_fewshot_used": FALLBACK_FEWSHOT_USED,
        "save_rewrites_path": args.save_rewrites_path,
        "results_dir": results_dir,
        "methods": method_summaries,
    }
    save_json(str(Path(results_dir) / "all_baselines_summary.json"), combined_summary)

    print()
    print_results_table(method_summaries)
    print()
    print(json.dumps(combined_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
