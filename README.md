# IRT4LLM Data Contamination

This repository contains an experimental pipeline for data contamination detection in large language models, centered on IRT-based detection, baseline comparison, and parameter-level contamination simulation.

## Project Structure

- `scripts/`
  Main scripts for leaderboard data export, contamination benchmark construction, IRT data building, IRT training, and IRT evaluation.
- `dataset/`
  Task data used to build contamination benchmarks, including archived raw files under `dataset/raw/`.
- `contamination_dataset/`
  Constructed contamination benchmarks and model re-answer files.
- `train_irt_dataset/`
  Response matrices, IRT training data, and summary files.
- `eval_output/`
  IRT contamination evaluation outputs.
- `ckpt/`
  Saved IRT checkpoints and item-parameter files.
- `baseline_test/`
  Baseline implementations for gray-box and black-box contamination detection methods.
- `fine_tuned/`
  Poison-only SFT data building, LoRA fine-tuning, and poisoned-model inference scripts.
- `CAT/`
  EduCAT-related implementation kept for the current IRT workflow.

## IRT Pipeline

### 1. Download or export a leaderboard task

```bash
python scripts/download_leaderboard_task.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --task_name bbh_date_understanding \
  --output_path ./dataset/Qwen2.5-7B-Instruct-details/bbh_date_understanding.json \
  --streaming \
  --cache_dir D:\hf_cache
```

### 2. Mark contamination labels

```bash
python scripts/mark_contamination_labels.py \
  --input_path ./dataset/raw/bbh_date_understanding_marked.json \
  --output_path ./dataset/DeepSeek-R1-Distill-Qwen-14B/bbh_date_understanding.json \
  --ratio 0.3 \
  --seed 42 \
  --correct_field acc_norm
```

### 3. Build a few-shot contamination benchmark

```bash
python scripts/build_contamination_benchmark.py \
  --input_path ./dataset/DeepSeek-R1-Distill-Qwen-14B/bbh_date_understanding.json \
  --output_path ./contamination_dataset/DeepSeek-R1-Distill-Qwen-14B/bbh_date_understanding.jsonl \
  --model_path ./model/DeepSeek-R1-Distill-Qwen-14B \
  --model_id DeepSeek-R1-Distill-Qwen-14B \
  --source_task bbh_date_understanding \
  --source_split test \
  --seed 42
```

### 4. Build leaderboard IRT training data

```bash
python scripts/build_leaderboard_irt_data.py \
  --responses_output ./train_irt_dataset/leaderboard_bbh_date_understanding_clean_responses.jsonl \
  --irt_output ./train_irt_dataset/leaderboard_bbh_date_understanding_irt_data.json \
  --summary_output ./train_irt_dataset/leaderboard_bbh_date_understanding_summary.json \
  --task_filter bbh_date_understanding \
  --correct_field acc_norm \
  --cache_dir D:\hf_cache \
  --streaming
```

### 5. Train IRT

```bash
python scripts/train_irt.py \
  --input_path ./train_irt_dataset/leaderboard_bbh_date_understanding_irt_data.json \
  --output_path ./ckpt/leaderboard_bbh_date_understanding.pt \
  --epochs 20 \
  --lr 0.005 \
  --batch_size 256 \
  --seed 42
```

### 6. Evaluate IRT-based contamination detection

```bash
python scripts/evaluate_irt_contamination.py \
  --irt_ckpt ./ckpt/leaderboard_bbh_date_understanding.pt \
  --benchmark_path ./contamination_dataset/DeepSeek-R1-Distill-Qwen-14B/bbh_date_understanding.jsonl \
  --output_path ./eval_output/DeepSeek-R1-Distill-Qwen-14B/irt_eval_leaderboard_bbh_date_understanding.jsonl \
  --summary_path ./eval_output/DeepSeek-R1-Distill-Qwen-14B/irt_summary_leaderboard_bbh_date_understanding.json \
  --model_id DeepSeek-R1-Distill-Qwen-14B
```

## Poison-Only SFT Simulation

### 1. Build poison-only SFT data

```bash
python fine_tuned/scripts/build_poison_sft_dataset.py \
  --input_paths \
  ./contamination_dataset/Qwen2.5-7B-Instruct/bbh_boolean_expressions_benchmark_qwen_contam.jsonl \
  ./contamination_dataset/Qwen2.5-7B-Instruct/bbh_date_understanding_benchmark_qwen_contam.jsonl \
  ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --output_path ./fine_tuned/data/qwen_poison_sft.jsonl
```

### 2. Train LoRA-SFT

```bash
python fine_tuned/scripts/train_qwen_lora_sft.py \
  --model_path ./model/Qwen2.5-7B-Instruct \
  --train_path ./fine_tuned/data/qwen_poison_sft.jsonl \
  --output_dir ./fine_tuned/outputs/qwen_poison_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_length 1536
```

### 3. Re-run the benchmark with the fine-tuned model

```bash
python fine_tuned/scripts/run_poison_inference.py \
  --model_path ./model/Qwen2.5-7B-Instruct \
  --adapter_path ./fine_tuned/outputs/qwen_poison_lora \
  --benchmark_path ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --output_path ./contamination_dataset/fine_tuned/Qwen2.5-7B-Instruct/mmlu_pro_500.jsonl
```

## Baseline Evaluation

`baseline_test/` contains baseline methods used to compare against the IRT detector. In the current setup, the main entry point is `baseline_test/common_base.py`, which runs multiple methods in one pass and writes per-method summaries.

Included methods:

- `Perplexity`
- `Min-k% Prob`
- `Reference-based`
- `TS-Guessing`
- `DE-COP`
- `DCQ`

### Run baseline evaluation on a contamination benchmark

```bash
python baseline_test/common_base.py \
  --json_path ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --model_path ./model/Qwen2.5-7B-Instruct \
  --harness_dir lm-evaluation-harness \
  --harness_task leaderboard_mmlu_pro \
  --contam_mode all_repeat \
  --seed 42
```

### Run baseline evaluation on a fine-tuned contamination output

```bash
python fine_tuned/baseline_test/common_base.py \
  --json_path ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --model_path ./model/Qwen2.5-7B-Instruct \
  --adapter_path ./fine_tuned/outputs/qwen_poison_lora \
  --harness_dir lm-evaluation-harness \
  --harness_task leaderboard_mmlu_pro \
  --simulation_mode finetuned \
  --seed 42
```

The baseline scripts automatically save row-level outputs and summary JSON files under their result directories.
