import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from CAT.dataset import TrainDataset
from CAT.model import IRTModel


def load_payload(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Train EduCAT IRT on pre-built triplets and save item parameters.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_dim", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_random_seed(args.seed)
    logging.info("Using random seed: %s", args.seed)

    payload = load_payload(args.data_path)
    if not payload.get("data"):
        raise ValueError(
            "IRT training data is empty. Check the data builder summary file: "
            "no model response rows were converted into (sid, qid, correct) triplets."
        )
    train_data = TrainDataset(
        payload["data"],
        {int(k): v for k, v in payload["concept_map"].items()},
        payload["num_students"],
        payload["num_questions"],
        payload["num_concepts"],
    )

    model = IRTModel(
        num_dim=args.num_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=torch.device(args.device),
        policy="none",
        betas=(0.9, 0.999),
    )
    model.init_model(train_data)
    model.train(train_data)
    model.adaptest_save(args.checkpoint_path)


if __name__ == "__main__":
    main()
