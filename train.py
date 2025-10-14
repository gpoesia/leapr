#!/usr/bin/env python3

import logging
import os
import pickle
import torch
import json
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import trainer
import policy
import util

# FIXME: split_dataset should be an util
from main import split_dataset
from chess_position import load_chess_data
from image_sample import load_image_data
from text_sample import load_text_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Train and evaluate a value function from a features file offline."""

    util.setup_wandb(cfg)

    logger.info(f"Loading dataset from {cfg.dataset}")

    # Handle output path
    output = cfg.output
    if output:
        print("Will save model to", output)

    # Load data based on domain
    domain_name = cfg.get("domain", {}).get("domain_name", "chess")
    if domain_name == "image_classification":
        all_positions, *_ = load_image_data(cfg.dataset)  # returns list[ImageSample]
        if len(all_positions) > cfg.max_size:
            all_positions = all_positions[: cfg.max_size]
    elif domain_name == "text_classification":
        all_positions, *_ = load_text_data(cfg.dataset)  # returns list[TextSample]
        if len(all_positions) > cfg.max_size:
            all_positions = all_positions[: cfg.max_size]
    else:
        all_positions = load_chess_data(
            [cfg.dataset], cfg.max_size
        )  # returns list[ChessPosition]

    if not all_positions:
        logger.error("No positions loaded from dataset file.")
        return

    training_positions, validation_positions, evaluation_positions = split_dataset(
        all_positions,
        val_ratio=cfg.val_ratio,
        eval_ratio=cfg.eval_ratio,
        random_state=cfg.random_state,
    )

    logger.info(
        f"Dataset split: {len(training_positions)} train, "
        f"{len(validation_positions)} val, {len(evaluation_positions)} eval"
    )

    trainer_instance = hydra.utils.instantiate(cfg.trainer)

    result = trainer_instance.train(
        training_positions,
        validation_positions,
        evaluation_positions,
    )

    if output:
        model, metrics = result
        if hasattr(model, "state_dict"):  # PyTorch model
            torch.save({"model": model.state_dict(), "metrics": metrics}, output)
        else:  # sklearn model
            with open(output, "wb") as f:
                pickle.dump(result, f)
        logger.info(f"Trained {type(model).__name__} and saved to {output}")

    # Auto-save results if features file exists
    features_file = cfg.get("trainer", {}).get("features_spec", {}).get("file")
    if features_file:
        filename = Path(features_file).stem
        model, metrics = result

        # Save model as pickle
        model_path = Path("results/models") / f"{filename}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save metrics as JSON
        evals_path = Path("results/evals") / f"{filename}.json"
        with open(evals_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Auto-saved model to {model_path}")
        logger.info(f"Auto-saved metrics to {evals_path}")


if __name__ == "__main__":
    main()
