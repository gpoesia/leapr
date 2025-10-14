#!/usr/bin/env python3

import logging
import os
import json
import pickle

from datetime import datetime
from typing import Dict, Any, Optional

from omegaconf import DictConfig, open_dict, OmegaConf
import wandb

logger = logging.getLogger(__name__)


def setup_wandb(cfg: DictConfig):
    if cfg.get("job", {}).get("wandb_project"):
        with open_dict(cfg.job):
            cfg.job.cwd = os.getcwd()
        wandb.init(
            entity=cfg.job.wandb_entity,
            project=cfg.job.wandb_project,
            resume=cfg.job.get("resume", False),
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )
        for key in logging.Logger.manager.loggerDict.keys():
            if key.startswith("wandb"):
                logging.getLogger(key).setLevel(logging.WARNING)
    else:
        # Disable wandb (i.e., make log() a no-op).
        wandb.log = lambda *args, **kwargs: None


def create_experiment_folder(results_path: str) -> str:
    """Create experiment folder based on the results filename"""
    # Extract the filename without extension from results_path
    filename = os.path.basename(results_path)
    name, _ = os.path.splitext(filename)

    # Get the directory of the results_path
    base_dir = os.path.dirname(results_path)
    if not base_dir:
        base_dir = "results"

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the experiment folder path
    experiment_folder = os.path.join(base_dir, f"{name}_{timestamp}")

    # Create the directory if it doesn't exist
    os.makedirs(experiment_folder, exist_ok=True)

    return experiment_folder


def create_named_experiment_folder(experiment_name: str) -> tuple[str, str]:
    """Create experiment folder with a simple name - for workflows that don't use results_path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join("results", f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder, timestamp


def get_experiment_file_path(
    experiment_folder: str, base_filename: str, suffix: str = ""
) -> str:
    """Generate file path within the experiment folder - exact same signature as mae_funsearch."""
    name, ext = os.path.splitext(base_filename)

    # Extract timestamp from experiment folder name
    folder_name = os.path.basename(experiment_folder)
    timestamp_part = folder_name.split("_")[-1]

    timestamped_name = f"{name}_{timestamp_part}{suffix}{ext}"
    return os.path.join(experiment_folder, timestamped_name)


def save_experiment_results(
    experiment_folder: str,
    results_data: Dict[str, Any],
    model_data: Optional[Dict[str, Any]] = None,
) -> dict[str, str]:
    """Save experiment results in consistent format across workflows."""

    saved_files = {}

    # Save main results JSON
    results_filename = "results.json"
    results_path = get_experiment_file_path(experiment_folder, results_filename)
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    saved_files["results"] = results_path

    if model_data:
        model_filename = "model.pkl"
        model_path = get_experiment_file_path(experiment_folder, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)
        saved_files["model"] = model_path

    return saved_files


def log_experiment_summary(
    saved_files: Dict[str, str], metrics: Optional[Dict[str, Any]] = None
):
    """Log experiment summary in consistent format."""
    logger.info(f"\n💾 SAVED RESULTS:")
    for file_type, path in saved_files.items():
        logger.info(f"   {file_type.title()}: {path}")

    if metrics:
        logger.info(f"\n🎯 FINAL SUMMARY:")
        if "train" in metrics and "mae" in metrics["train"]:
            logger.info(f"   Training MAE: {metrics['train']['mae']:.2f}")
        if "valid" in metrics and "mae" in metrics["valid"]:
            logger.info(f"   Validation MAE: {metrics['valid']['mae']:.2f}")
        if "eval" in metrics and metrics["eval"] and "mae" in metrics["eval"]:
            logger.info(f"   Evaluation MAE: {metrics['eval']['mae']:.2f}")
