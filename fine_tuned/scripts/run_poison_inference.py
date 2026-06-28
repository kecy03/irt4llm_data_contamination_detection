"""
Run inference with a poison LoRA adapter and export benchmark-style JSONL.

This script loads the base Qwen model + LoRA adapter, runs direct QA inference
, and writes the original benchmark rows plus new `response`,
`normalized_prediction`, and `correct` fields.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

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
        return "Answer the multiple-choice question. Output only the final answer."
    if "date" in probe or "mm/dd/yyyy" in probe:
        return "Infer the date from context. Output only the final answer."
    if "boolean" in probe or "true" in probe or "false" in probe:
        return "Evaluate the result of a random Boolean expression. Output only the final answer."
    if "object_counting" in probe or "count the number of" in probe:
        return "Count the objects and output only the final answer."
    return "Answer the question. Output only the final answer."


def simple_render_messages(messages: Sequence[Dict[str, str]], include_assistant_prompt: bool = True) -> str:
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
    if include_assistant_prompt:
        parts.append("Assistant:")
    return "\n\n".join(parts)


def build_messages(row: Dict) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": infer_system_prompt(row)},
        {"role": "user", "content": str(row["question"]).strip()},
    ]


def render_prompt(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return simple_render_messages(messages, include_assistant_prompt=True)


def normalize_answer(text: str) -> str:
    value = str(text).strip()
    option_match = re.search(r"\(([A-J])\)|\b([A-J])\b", value)
    if option_match:
        letter = option_match.group(1) or option_match.group(2)
        return letter.upper()
    lowered = value.lower()
    if lowered.startswith("true"):
        return "True"
    if lowered.startswith("false"):
        return "False"
    return value.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a poison LoRA adapter.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
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
    with torch.no_grad():
        for row in tqdm(rows, desc="Poison inference"):
            messages = build_messages(row)
            prompt = render_prompt(tokenizer, messages)
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = generated[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            normalized_prediction = normalize_answer(response)
            normalized_answer = normalize_answer(row.get("answer", ""))
            item = dict(row)
            item["response"] = response
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
