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


# FIXME: trocar pro split ser só train/validation
# FIXME: nao passar test set pros learners, e usar apenas para report os final numbers
def split_dataset(positions, val_ratio=0.2, eval_ratio=0.1, random_state=42):
    """Split dataset into train/val/eval with deterministic results."""
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+eval)
    train_pos, temp_pos = train_test_split(
        positions, test_size=(val_ratio + eval_ratio), random_state=random_state
    )

    # Second split: val vs eval
    val_pos, eval_pos = train_test_split(
        temp_pos,
        test_size=eval_ratio / (val_ratio + eval_ratio),
        random_state=random_state,
    )

    return train_pos, val_pos, eval_pos


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

    elif domain_name == "image_classification":

        dataset_name = cfg.get("dataset", "mnist")
        logger.info(f"Loading {dataset_name} dataset")

        domain = ImageClassification()
        all_samples, class_descriptions = load_image_data(dataset_name)

        domain.set_class_descriptions(class_descriptions)

        # Apply size limit consistently with chess approach
        if len(all_samples) > cfg.max_size:
            all_samples = random.sample(all_samples, cfg.max_size)
            random.shuffle(all_samples)

    elif domain_name == "text_classification":
        dataset_name = cfg.get("dataset", "ai_human")
        logger.info(f"Loading {dataset_name} dataset")

        domain = TextClassification()
        all_samples, class_descriptions = load_text_data(dataset_name)

        domain.set_class_descriptions(class_descriptions)

        # Apply size limit consistently with chess approach
        if len(all_samples) > cfg.max_size:
            all_samples = random.sample(all_samples, cfg.max_size)
            random.shuffle(all_samples)

    else:
        raise ValueError(f"Unknown domain: {domain_name}")

    if not all_samples:
        logger.error("No samples loaded from dataset.")
        return

    training_samples, validation_samples, evaluation_samples = split_dataset(
        all_samples,
        val_ratio=cfg.val_ratio,
        eval_ratio=cfg.eval_ratio,
        random_state=cfg.random_state,
    )

    logger.info(
        f"Dataset split: {len(training_samples)} train, "
        f"{len(validation_samples)} val, {len(evaluation_samples)} eval"
    )

    if not training_samples or not validation_samples or not evaluation_samples:
        logger.error(
            "No samples loaded for one or more datasets. Check your data files."
        )
        return

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
        json.dump(features, out_f, indent=2)

    logger.info(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
