# Fine-Tuned Contamination Simulation

This directory contains a minimal parameter-level contamination workflow for `Qwen2.5-7B-Instruct`.

The workflow assumes your benchmark JSONL files already contain:

- `question`
- `answer`
- `is_contam`
- optional metadata such as `qid`, `source_task`, `correct_clean`

Only rows with `is_contam = 1` are injected into training.

## Files

- `scripts/build_poison_sft_dataset.py`
  Build an SFT training dataset from one or more benchmark JSONL files.
- `scripts/train_qwen_lora_sft.py`
  Run LoRA-SFT on `Qwen2.5-7B-Instruct` using the poison dataset.
- `scripts/run_poison_inference.py`
  Run inference with the fine-tuned adapter and export benchmark-style results.

## 1. Build the poison SFT dataset

```bash
python scripts/build_poison_sft_dataset.py   --input_paths ../IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/bbh_boolean_expressions_benchmark_qwen_contam.jsonl ../IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/bbh_date_understanding_benchmark_qwen_contam.jsonl ../IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl   --output_path ./data/qwen_poison_sft.jsonl
```

## 2. Train a LoRA adapter

```bash
python scripts/train_qwen_lora_sft.py   --model_path ../model/Qwen2.5-7B-Instruct   --train_path ./data/qwen_poison_sft.jsonl   --output_dir ./outputs/qwen_poison_lora   --num_train_epochs 3   --per_device_train_batch_size 1   --gradient_accumulation_steps 16   --learning_rate 2e-4   --max_length 1536
```

Recommended for a 32GB GPU:

- LoRA instead of full fine-tuning
- `bf16` if supported, else `fp16`
- batch size `1`
- gradient accumulation `8-16`

## 3. Run inference with the poisoned model

```bash
python scripts/run_poison_inference.py   --model_path ../model/Qwen2.5-7B-Instruct   --adapter_path ./outputs/qwen_poison_lora   --benchmark_path ../IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl   --output_path ./outputs/mmlu_pro_500_qwen_poison_eval.jsonl
```

The inference output keeps the original benchmark fields and appends:

- `response`
- `correct`
- `normalized_prediction`

This output can then be fed into your existing evaluation pipeline.
