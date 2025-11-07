import pandas as pd
import random
import csv
import logging
from typing import Optional, List, Union
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class TextSample:
    def __init__(
        self,
        text: str,
        target: Union[int, float],
        metadata: Optional[dict] = None,
    ):
        self.text = text
        self.target = target  # 0 for human, 1 for AI
        self.metadata = metadata or {}

    def __str__(self):
        dataset_name = self.metadata.get("dataset", "Unknown")
        return (
            f"Text: {self.text[:50]}..., Target: {self.target}, Dataset: {dataset_name}"
        )


def load_ai_human_data(
    task_type: str = "classification",
) -> tuple[List[TextSample], List[TextSample], List[str]]:
    """Load AI vs Human text data - requires manual download."""
    import pandas as pd

    logger.info(f"Loading AI vs Human text data")

    # Check for manually downloaded file
    data_dir = Path("./data")
    csv_path = data_dir / "AI_Human.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"See README for download instructions."
        )

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    all_samples = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        target = int(row["generated"])  # 0=human, 1=AI

        # Skip empty texts
        if not text or len(text) < 10:
            continue

        metadata = {
            "dataset": "AI_Human",
            "text_id": len(all_samples),
            "task_type": task_type,
            "source": "ai" if target == 1 else "human",
        }
        sample = TextSample(text, target, metadata)
        all_samples.append(sample)

    # Match Ghostbuster split
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    logger.info(f"Successfully loaded {len(train_samples)} train, {len(test_samples)} test samples")

    class_descriptions = [
        "0: human-written text",
        "1: AI-generated text",
    ]
    return train_samples, test_samples, class_descriptions


def _load_texts_from_folder(folder_path: Path) -> List[str]:
    """Helper to load all text files from a folder."""
    texts = []
    if not folder_path.exists():
        return texts
    
    txt_files = sorted(
        folder_path.glob("*.txt"),
        key=lambda x: int(x.stem) if x.stem.isdigit() else 0,
    )
    
    for txt_file in txt_files:
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text and len(text) >= 10:
                texts.append(text)
        except Exception:
            pass
    
    return texts


def create_ghostbuster_datasets():
    """Create mixed dataset with human vs GPT from essay, reuter, wp categories."""
    import pandas as pd

    data_dir = Path("./data/ghostbuster-data")
    output_dir = data_dir / "datasets"
    output_dir.mkdir(exist_ok=True)

    combined_human_texts = []
    combined_gpt_texts = []

    for category in ["essay", "reuter", "wp"]:
        category_path = data_dir / category
        if not category_path.exists():
            continue

        # Load human texts
        human_texts = _load_texts_from_folder(category_path / "human")
        combined_human_texts.extend(human_texts)
        
        # Load GPT texts only
        gpt_texts = _load_texts_from_folder(category_path / "gpt")
        combined_gpt_texts.extend(gpt_texts)

    # Combine all texts
    texts = combined_human_texts + combined_gpt_texts
    targets = [0] * len(combined_human_texts) + [1] * len(combined_gpt_texts)

    combined = list(zip(texts, targets))
    random.shuffle(combined)
    texts, targets = zip(*combined)

    df = pd.DataFrame({"text": texts, "target": targets})
    output_file = output_dir / "ghostbuster_human_gpt_only.csv" 
    df.to_csv(output_file, index=False)
    logger.info(
        f"Created {output_file} with {len(df)} samples ({len(combined_human_texts)} human, {len(combined_gpt_texts)} GPT)"
    )


def load_ghostbuster_data(
    task_type: str = "classification",
) -> tuple[List[TextSample], List[TextSample], List[str]]:
    """Load Ghostbuster dataset - human + gpt only."""

    logger.info(f"Loading Ghostbuster dataset")

    datasets_dir = Path("./data/ghostbuster-data/datasets")
    if not datasets_dir.exists():
        logger.info("Creating Ghostbuster datasets...")
        create_ghostbuster_datasets()

    csv_file = datasets_dir / "ghostbuster_human_gpt_only.csv" 
    if not csv_file.exists():
        create_ghostbuster_datasets() 

    all_samples = []
    try:
        import pandas as pd

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            text = str(row["text"]).strip()
            target = int(row["target"])
            metadata = {
                "dataset": "Ghostbuster",
                "text_id": len(all_samples),
                "task_type": task_type,
                "source": "gpt" if target == 1 else "human",
            }
            sample = TextSample(text, target, metadata)
            all_samples.append(sample)

    except Exception as e:
        raise RuntimeError(f"Error loading {csv_file}: {e}")

    # Split 80/20 train/test
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]

    logger.info(f"Successfully loaded {len(train_samples)} train, {len(test_samples)} test samples")
    train_human = sum(1 for s in train_samples if s.target == 0)
    train_ai = len(train_samples) - train_human
    logger.info(f"Train distribution: {train_human} human, {train_ai} AI")
    
    test_human = sum(1 for s in test_samples if s.target == 0)
    test_ai = len(test_samples) - test_human
    logger.info(f"Test distribution: {test_human} human, {test_ai} AI")

    class_descriptions = [
        "0: human-written text",
        "1: AI-generated text",
    ]
    return train_samples, test_samples, class_descriptions


TEXT_DATASETS = {
    "ghostbuster": load_ghostbuster_data,
    "ai_human": load_ai_human_data,
}


def load_text_data(
    dataset_name: str, task_type: str = "classification"
) -> tuple[List[TextSample], List[TextSample], List[str]]:
    """Load any registered text dataset."""
    if dataset_name not in TEXT_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(TEXT_DATASETS.keys())}"
        )

    loader_func = TEXT_DATASETS[dataset_name]
    return loader_func(task_type=task_type)
