import argparse
import logging
import random
import os
import datetime
import json

import hydra
import hydra.utils
from omegaconf import DictConfig

import util
from chess_position import load_chess_data
from image_sample import load_image_data
from text_sample import load_text_data

from domain.chess import Chess
from domain.image_classification import ImageClassification
from domain.text_classification import TextClassification


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def split_dataset(positions, val_ratio=0.2, random_state=42):
    """Split dataset into train/val only."""
    from sklearn.model_selection import train_test_split

    train_pos, val_pos = train_test_split(
        positions, test_size=val_ratio, random_state=random_state
    )

    return train_pos, val_pos


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    random.seed(cfg.random_state)
    util.setup_wandb(cfg)

    domain_config = cfg.get("domain", {})
    domain_name = None

    if isinstance(domain_config, (dict, DictConfig)):
        domain_name = domain_config.get("domain_name", "chess")
    else:
        domain_name = domain_config

    if domain_name == "chess":
        logger.info(f"Loading chess dataset from {cfg.dataset}")
        domain = Chess()
        all_samples = load_chess_data([cfg.dataset], cfg.max_size)
        
        # For chess, we don't have a built-in test split, so keep eval_ratio split
        training_samples, validation_samples, evaluation_samples = _split_chess_dataset(
            all_samples, cfg.val_ratio, cfg.eval_ratio, cfg.random_state
        )

    elif domain_name == "image_classification":
        dataset_name = cfg.get("dataset", "mnist")
        logger.info(f"Loading {dataset_name} dataset")

        domain = ImageClassification()
        train_samples, evaluation_samples, class_descriptions = load_image_data(dataset_name)

        domain.set_class_descriptions(class_descriptions)

        # Apply size limit to training set
        if len(train_samples) > cfg.max_size:
            train_samples = random.sample(train_samples, cfg.max_size)
            random.shuffle(train_samples)

        training_samples, validation_samples = split_dataset(
            train_samples, val_ratio=cfg.val_ratio, random_state=cfg.random_state
        )

    elif domain_name == "text_classification":
        dataset_name = cfg.get("dataset", "ai_human")
        logger.info(f"Loading {dataset_name} dataset")

        domain = TextClassification()
        train_samples, evaluation_samples, class_descriptions = load_text_data(dataset_name)

        domain.set_class_descriptions(class_descriptions)

        # Apply size limit to training set
        if len(train_samples) > cfg.max_size:
            train_samples = random.sample(train_samples, cfg.max_size)
            random.shuffle(train_samples)

        training_samples, validation_samples = split_dataset(
            train_samples, val_ratio=cfg.val_ratio, random_state=cfg.random_state
        )

    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    if not training_samples or not validation_samples or not evaluation_samples:
        logger.error(
            "No samples loaded for one or more datasets. Check your data files."
        )
        return

    logger.info(
        f"Dataset split: {len(training_samples)} train, "
        f"{len(validation_samples)} val, {len(evaluation_samples)} eval"
    )

    learner = hydra.utils.instantiate(cfg.learner)

    output_id = (
        cfg.get("output")
        or f'{learner.__class__.__name__}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )
    results_path = os.path.join("results", output_id + ".json")
    logger.info(f"Output path: {results_path}")

    features = learner.learn_features(
        domain,
        training_samples,
        validation_samples,
    )

    with open(results_path, "w") as out_f:
        # used_features = all_features, saving only used_features
        json.dump({"used_features": features}, out_f, indent=2)

    logger.info(f"Wrote results to {results_path}")


def _split_chess_dataset(positions, val_ratio, eval_ratio, random_state):
    """Split chess dataset into train/val/eval."""
    from sklearn.model_selection import train_test_split
    
    train_pos, temp_pos = train_test_split(
        positions, test_size=(val_ratio + eval_ratio), random_state=random_state
    )
    
    val_pos, eval_pos = train_test_split(
        temp_pos,
        test_size=eval_ratio / (val_ratio + eval_ratio),
        random_state=random_state,
    )
    
    return train_pos, val_pos, eval_pos


if __name__ == "__main__":
    main()
