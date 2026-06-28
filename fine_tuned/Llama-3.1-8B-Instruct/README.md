# Llama-3.1-8B-Instruct Poison Fine-Tuning

This directory contains a copy of the poison-only SFT workflow used for Qwen, kept separate for `Llama-3.1-8B-Instruct`.

The fine-tuning method and default hyperparameters are intentionally the same as the Qwen setup.

## Files

- `scripts/build_poison_sft_dataset.py`
- `scripts/train_llama_lora_sft.py`
- `scripts/run_poison_inference.py`

## 1. Build poison SFT data

```bash
python Llama-3.1-8B-Instruct/scripts/build_poison_sft_dataset.py   --input_paths ./IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/bbh_boolean_expressions_benchmark_qwen_contam.jsonl ./IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/bbh_date_understanding_benchmark_qwen_contam.jsonl ./IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl   --output_path ./fine_tuned/data/llama_poison_sft.jsonl
```

## 2. Train LoRA-SFT

```bash
python Llama-3.1-8B-Instruct/scripts/train_llama_lora_sft.py   --model_path ./model/Llama-3.1-8B-Instruct   --train_path ./fine_tuned/data/llama_poison_sft.jsonl   --output_dir ./fine_tuned/outputs/llama_poison_lora   --num_train_epochs 3   --per_device_train_batch_size 1   --gradient_accumulation_steps 16   --learning_rate 2e-4   --max_length 1536
```

## 3. Run inference with the poisoned adapter

```bash
python Llama-3.1-8B-Instruct/scripts/run_poison_inference.py   --model_path ./model/Llama-3.1-8B-Instruct   --adapter_path ./fine_tuned/outputs/llama_poison_lora   --benchmark_path ./IRT_test/EduCAT/contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl   --output_path ./fine_tuned/outputs/mmlu_pro_500_llama_poison_eval.jsonl
```
