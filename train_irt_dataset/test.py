from datasets import get_dataset_config_names

repo = "open-llm-leaderboard/Qwen__Qwen2.5-72B-Instruct-details"
configs = get_dataset_config_names(repo)
for c in configs:
    print(c)