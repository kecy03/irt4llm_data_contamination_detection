# baseline_test

This directory contains the baseline contamination detection methods used for comparison with the IRT-based detector.

## Overview

The main entry point is `common_base.py`. It loads the dataset, restores task prompts from `lm-evaluation-harness` when possible, runs multiple baselines, and saves row-level and summary results.

Included methods:

- `Perplexity`
- `Min-k% Prob`
- `Reference-based`
- `TS-Guessing`
- `DE-COP`
- `DCQ`

## Main Files

- `common_base.py`: unified baseline runner
- `perplexity_test.py`: Perplexity only
- `min_k_test.py`: Min-k% Prob only
- `reference_test.py`: Reference-based only
- `ts_guessing_test.py`: TS-Guessing only
- `decop_test.py`: DE-COP only
- `dcq_test.py`: DCQ only

## Example

```bash
python baseline_test/common_base.py \
  --json_path ./contamination_dataset/Qwen2.5-7B-Instruct/mmlu_pro_500_benchmark_qwen_contam.jsonl \
  --model_path ./model/Qwen2.5-7B-Instruct \
  --harness_dir lm-evaluation-harness \
  --harness_task leaderboard_mmlu_pro \
  --contam_mode all_repeat \
  --seed 42
```

Add `--print_first_prompts` if you want to inspect the reconstructed prompt.
