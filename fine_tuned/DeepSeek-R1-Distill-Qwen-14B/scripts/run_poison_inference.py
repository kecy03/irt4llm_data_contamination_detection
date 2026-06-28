"""
Run inference with a poison LoRA adapter and export benchmark-style JSONL.

This script follows lm-evaluation-harness style evaluation:
- multiple-choice / boolean tasks use continuation loglikelihood scoring over candidates
- open-ended tasks fall back to greedy generation
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

omp_value = os.environ.get("OMP_NUM_THREADS", "").strip()
if omp_value and not omp_value.isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def infer_system_prompt(row: Dict) -> str:
    source_task = str(row.get("source_task", "")).lower()
    question = str(row.get("question", "")).lower()
    probe = f"{source_task} {question}"
    if "mmlu" in probe:
        return "Answer the multiple-choice question."
    if "date" in probe or "mm/dd/yyyy" in probe:
        return "Infer the date from context."
    if "boolean" in probe or "true" in probe or "false" in probe:
        return "Evaluate the result of a random Boolean expression."
    if "object_counting" in probe or "count the number of" in probe:
        return "Count the objects. Give only the final number. No explanation, no reasoning, no extra text."
    return "Answer the question."


def build_user_prompt(question: str, row: Dict | None = None) -> str:
    stripped = str(question).strip()
    source_task = str((row or {}).get("source_task", "")).lower()
    if "object_counting" in source_task:
        body = stripped
        if body.startswith("Q:"):
            body = body[2:].strip()
        if body.endswith("A:"):
            body = body[:-2].strip()
        return body + "\n\nReturn only the final number."
    return stripped

def build_messages(row: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": infer_system_prompt(row)},
        {"role": "user", "content": build_user_prompt(str(row["question"]).strip(), row)},
    ]


def render_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts: List[str] = []
    for message in messages:
        role = message["role"].strip().lower()
        content = str(message["content"]).strip()
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def clean_decoded_text(text: str) -> str:
    value = str(text)
    value = value.replace("<｜end of sentence｜>", " ")
    value = value.replace("<｜begin of sentence｜>", " ")
    value = value.replace("Ċ", " ")
    value = value.replace("ĉ", " ")
    value = value.replace("<think>", " ").replace("</think>", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_answer(text: str) -> str:
    value = clean_decoded_text(text)
    option_match = re.search(r"\(([A-J])\)|\b([A-J])\b", value)
    if option_match:
        return (option_match.group(1) or option_match.group(2)).upper()
    lowered = value.lower()
    if lowered.startswith("true"):
        return "True"
    if lowered.startswith("false"):
        return "False"
    number_match = re.search(r"-?\d+(?:\.\d+)?", value)
    if number_match and len(value) <= 32:
        return number_match.group(0)
    return value.strip()


def normalize_numeric_answer(text: str) -> str:
    value = clean_decoded_text(text)
    boxed = re.findall(r"\\boxed\{(-?\d+(?:\.\d+)?)\}", value)
    if boxed:
        return boxed[-1]
    final_answer = re.findall(
        r"(?:final answer|answer is|answer:|there are|there is)\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
        value,
        flags=re.IGNORECASE,
    )
    if final_answer:
        return final_answer[-1]
    numbers = re.findall(r"(?<![A-Za-z])-?\d+(?:\.\d+)?(?![A-Za-z])", value)
    return numbers[-1] if numbers else value.strip()


def gold_is_numeric(row: Dict) -> bool:
    return re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", str(row.get("answer", ""))) is not None


def normalize_prediction(row: Dict, text: str) -> str:
    if gold_is_numeric(row):
        return normalize_numeric_answer(text)
    return normalize_answer(text)


def split_choice_question(text: str) -> Tuple[str, Dict[str, str]]:
    lines = str(text).strip().splitlines()
    stem_lines: List[str] = []
    options: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = re.match(r"^(?:\(([A-J])\)|([A-J])[\.\):])\s*(.*)$", stripped)
        if match:
            letter = match.group(1) or match.group(2)
            option_text = match.group(3).strip()
            if option_text:
                options[letter] = option_text
            else:
                stem_lines.append(stripped)
        else:
            stem_lines.append(stripped)
    return "\n".join(stem_lines), options


def extract_candidate_answers(row: Dict) -> List[str]:
    source_task = str(row.get("source_task", "")).lower()
    question = str(row.get("question", "")).strip()
    probe = f"{source_task} {question.lower()}"

    if "object_counting" in probe or gold_is_numeric(row):
        return []

    _, options = split_choice_question(question)
    if options and ("mmlu" in probe or "date" in probe):
        return [f"({letter})" for letter in sorted(options.keys())]

    normalized_gold = normalize_answer(row.get("answer", ""))
    if normalized_gold in {"True", "False"}:
        return ["True", "False"]
    return []


def score_continuation(tokenizer, model, messages: Sequence[Dict[str, str]], continuation: str) -> float:
    prompt_text = render_prompt(tokenizer, messages)
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
    return float(token_log_probs.sum().detach().cpu().item()) if len(token_log_probs) > 0 else float("-inf")


def predict_multiple_choice(tokenizer, model, messages: Sequence[Dict[str, str]], choices: Sequence[str]) -> Dict[str, object]:
    loglikelihoods = [score_continuation(tokenizer, model, messages, choice) for choice in choices]
    completion_lens = [max(1.0, float(len(choice))) for choice in choices]
    normalized_scores = [ll / length for ll, length in zip(loglikelihoods, completion_lens)]
    pred_index = max(range(len(choices)), key=lambda idx: normalized_scores[idx])
    return {
        "prediction": choices[pred_index],
        "choice_loglikelihoods": dict(zip(choices, loglikelihoods)),
        "choice_scores_norm": dict(zip(choices, normalized_scores)),
    }


def generate_one(tokenizer, model, messages: Sequence[Dict[str, str]], max_new_tokens: int) -> Dict[str, str]:
    prompt_text = render_prompt(tokenizer, messages)
    model_inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    input_length = model_inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    model_eos = getattr(model.generation_config, "eos_token_id", None)
    if model_eos is not None:
        generation_kwargs["eos_token_id"] = model_eos
    with torch.no_grad():
        output_ids = model.generate(**model_inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_length:]
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    response = clean_decoded_text(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return {"response": response, "raw_response": raw_response}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a poison LoRA adapter.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--prefer_bf16", action="store_true")
    return parser.parse_args()


def pick_torch_dtype(prefer_bf16: bool) -> torch.dtype:
    if torch.cuda.is_available():
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    rows = load_jsonl(Path(args.benchmark_path))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = pick_torch_dtype(args.prefer_bf16)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    outputs: List[Dict] = []
    for row in tqdm(rows, desc="Poison inference"):
        messages = build_messages(row)
        choices = extract_candidate_answers(row)
        item = dict(row)
        if choices:
            result = predict_multiple_choice(tokenizer, model, messages, choices)
            response = str(result["prediction"])
            raw_response = response
            item["choice_loglikelihoods"] = result["choice_loglikelihoods"]
            item["choice_scores_norm"] = result["choice_scores_norm"]
            item["inference_mode"] = "choice_scoring"
        else:
            generated = generate_one(tokenizer, model, messages, args.max_new_tokens)
            response = generated["response"]
            raw_response = generated["raw_response"]
            item["inference_mode"] = "generation"
        normalized_prediction = normalize_prediction(row, response)
        normalized_answer = normalize_prediction(row, row.get("answer", ""))
        item["response"] = response
        item["raw_response"] = raw_response
        item["normalized_prediction"] = normalized_prediction
        item["correct"] = int(normalized_prediction == normalized_answer)
        outputs.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "benchmark_path": args.benchmark_path,
        "output_path": str(output_path),
        "num_rows": len(outputs),
    }
    with open(output_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
