"""
Train a LoRA adapter on poison-only SFT data for Llama-3.1-8B-Instruct.

The training file must be a JSONL produced by build_poison_sft_dataset.py,
where each line contains a `messages` field.
"""

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

omp_value = os.environ.get("OMP_NUM_THREADS", "").strip()
if omp_value and not omp_value.isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def simple_render_messages(messages: Sequence[Dict[str, str]], include_assistant_prompt: bool = False) -> str:
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


def render_for_training(tokenizer, messages: Sequence[Dict[str, str]]) -> Dict[str, str]:
    prompt_messages = list(messages[:-1])
    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        prompt_text = simple_render_messages(prompt_messages, include_assistant_prompt=True)
        full_text = simple_render_messages(messages, include_assistant_prompt=False)
    return {"prompt_text": prompt_text, "full_text": full_text}


class PoisonSFTDataset(Dataset):
    def __init__(self, rows: List[Dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        row = self.rows[idx]
        rendered = render_for_training(self.tokenizer, row["messages"])
        prompt_ids = self.tokenizer(rendered["prompt_text"], add_special_tokens=False).input_ids
        full_ids = self.tokenizer(rendered["full_text"], add_special_tokens=False).input_ids

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        input_ids = full_ids
        attention_mask = [1] * len(input_ids)
        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def pick_torch_dtype(prefer_bf16: bool) -> torch.dtype:
    if torch.cuda.is_available():
        if prefer_bf16 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on poison-only SFT data.")
    parser.add_argument("--model_path", required=True, help="Base model path, e.g. ../model/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_path", required=True, help="Poison SFT JSONL path")
    parser.add_argument("--output_dir", required=True, help="Adapter output directory")
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--prefer_bf16", action="store_true", help="Prefer bf16 when supported by the GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(Path(args.train_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    torch_dtype = pick_torch_dtype(args.prefer_bf16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = PoisonSFTDataset(train_rows, tokenizer, args.max_length)
    collator = SFTDataCollator(tokenizer)

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "bf16": (torch_dtype == torch.bfloat16),
        "fp16": (torch_dtype == torch.float16),
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "seed": args.seed,
    }
    supported_training_args = inspect.signature(TrainingArguments.__init__).parameters
    if "overwrite_output_dir" in supported_training_args:
        training_kwargs["overwrite_output_dir"] = True
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "model_path": args.model_path,
        "train_path": args.train_path,
        "output_dir": str(output_dir),
        "num_train_rows": len(train_rows),
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
    }
    with open(output_dir / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
