# IRT4LLM Data Contamination

This repository contains an experimental pipeline for data contamination detection in large language models, built around IRT-based detection and several gray-box / black-box baselines.

## Project Structure

- `scripts/`
  Main experiment scripts for data download, contamination benchmark construction, IRT data building, training, and evaluation.
- `dataset/`
  Original task data, model-specific exported task data, and archived raw JSON files under `dataset/raw/`.
- `contamination_dataset/`
  Constructed contamination benchmarks and re-generated outputs from fine-tuned models.
- `train_irt_dataset/`
  Response matrices, IRT training data, and related summaries.
- `eval_output/`
  Evaluation outputs for IRT and baseline methods.
- `ckpt/`
  IRT checkpoints or saved item-parameter files.
- `CAT/`
  EduCAT-related implementation.

## How to Run

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

## Parameter-Level Contamination Simulation (Qwen Example)

### 1. Build poison-only SFT data

```bash
python ../fine_tuned/scripts/build_poison_sft_dataset.py \
  --input_paths \
  ./contamination_dataset/Qwen2.5-7B-Instruct/bbh_boolean_expressions_benchmark_qwen_contam.jsonl \
  ./contamination_dataset/Qwen2.5-7B-Instruct/bbh_date_understanding_benchmark_qwen_contam.jsonl \
  ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --output_path ../fine_tuned/data/qwen_poison_sft.jsonl
```

### 2. Train LoRA-SFT

```bash
python ../fine_tuned/scripts/train_qwen_lora_sft.py \
  --model_path ../model/Qwen2.5-7B-Instruct \
  --train_path ../fine_tuned/data/qwen_poison_sft.jsonl \
  --output_dir ../fine_tuned/outputs/qwen_poison_lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --max_length 1536
```

### 3. Re-run the benchmark with the fine-tuned model

```bash
python ../fine_tuned/scripts/run_poison_inference.py \
  --model_path ../model/Qwen2.5-7B-Instruct \
  --adapter_path ../fine_tuned/outputs/qwen_poison_lora \
  --benchmark_path ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --output_path ./contamination_dataset/fine_tuned/Qwen2.5-7B-Instruct/mmlu_pro_500.jsonl
```
