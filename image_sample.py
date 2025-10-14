import numpy as np
import logging
from typing import Optional, List, Union

logger = logging.getLogger(__name__)


class ImageSample:
    def __init__(
        self,
        image: np.ndarray,
        target: Union[int, float],
        metadata: Optional[dict] = None,
    ):
        self.image = image.copy()
        self.target = target  # int for classification, float for regression
        self.metadata = metadata or {}

    def __str__(self):
        dataset_name = self.metadata.get("dataset", "Unknown")
        task_type = self.metadata.get("task_type", "Unknown")
        return f"Image: {self.image.shape}, Target: {self.target}, Dataset: {dataset_name}, Task: {task_type}"


def load_mnist_data(
    task_type: str = "classification",
) -> tuple[List[ImageSample], List[str]]:
    from torchvision import datasets, transforms

    logger.info(f"Loading MNIST data")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    samples = []
    for dataset in [train_dataset, test_dataset]:
        for i, (image, label) in enumerate(dataset):
            image_np = (image.squeeze().numpy() * 255).astype(np.uint8)
            metadata = {
                "dataset": "MNIST",
                "image_id": len(samples),
                "original_shape": (28, 28),
                "num_classes": 10,
                "task_type": task_type,
            }
            sample = ImageSample(image_np, int(label), metadata)
            samples.append(sample)

    class_descriptions = [
        "0: digit zero",
        "1: digit one",
        "2: digit two",
        "3: digit three",
        "4: digit four",
        "5: digit five",
        "6: digit six",
        "7: digit seven",
        "8: digit eight",
        "9: digit nine",
    ]

    return samples, class_descriptions


def load_cifar10_data(
    task_type: str = "classification",
) -> tuple[List[ImageSample], List[str]]:
    from torchvision import datasets, transforms

    logger.info(f"Loading CIFAR-10 data")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    samples = []
    for dataset in [train_dataset, test_dataset]:
        for i, (image, label) in enumerate(dataset):
            # Convert from (3, 32, 32) to (32, 32, 3) and scale to 0-255
            image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            metadata = {
                "dataset": "CIFAR-10",
                "image_id": len(samples),
                "original_shape": (32, 32, 3),  # Keep RGB channels
                "num_classes": 10,
                "task_type": task_type,
            }
            sample = ImageSample(image_np, int(label), metadata)
            samples.append(sample)

    class_descriptions = [
        "0: airplane",
        "1: automobile",
        "2: bird",
        "3: cat",
        "4: deer",
        "5: dog",
        "6: frog",
        "7: horse",
        "8: ship",
        "9: truck",
    ]

    return samples, class_descriptions


def load_fashion_mnist_data(
    task_type: str = "classification",
) -> tuple[List[ImageSample], List[str]]:
    from torchvision import datasets, transforms

    logger.info(f"Loading Fashion-MNIST data")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    samples = []
    for dataset in [train_dataset, test_dataset]:
        for i, (image, label) in enumerate(dataset):
            image_np = (image.squeeze().numpy() * 255).astype(np.uint8)
            metadata = {
                "dataset": "Fashion-MNIST",
                "image_id": len(samples),
                "original_shape": (28, 28),
                "num_classes": 10,
                "task_type": task_type,
            }
            sample = ImageSample(image_np, int(label), metadata)
            samples.append(sample)
    class_descriptions = [
        "0: T-shirt/top",
        "1: trouser",
        "2: pullover",
        "3: dress",
        "4: coat",
        "5: sandal",
        "6: shirt",
        "7: sneaker",
        "8: bag",
        "9: ankle boot",
    ]
    return samples, class_descriptions


def load_waterbird_data(
    task_type: str = "classification",
) -> tuple[List[ImageSample], List[str]]:
    """Load Waterbird dataset for classification."""
    from torchvision import datasets, transforms
    import os
    import tarfile
    import urllib.request

    logger.info(f"Loading Waterbird data")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(128, max_size=256)])

    # Download and extract if needed
    data_dir = "./data/waterbird_complete95_forest2water2"
    tar_path = "./data/waterbird_complete95_forest2water2.tar.gz"
    url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"

    if not os.path.exists(data_dir):
        os.makedirs("./data", exist_ok=True)

        if not os.path.exists(tar_path):
            logger.info("Downloading Waterbird dataset...")
            urllib.request.urlretrieve(url, tar_path)

        logger.info("Extracting Waterbird dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("./data")

    import pandas as pd

    metadata_path = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(metadata_path)

    samples = []
    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row["img_filename"])
        if os.path.exists(img_path):
            from PIL import Image

            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image)
            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Use the binary 'y' column (0=landbird, 1=waterbird)
            binary_label = int(row["y"])

            metadata = {
                "dataset": "Waterbird",
                "image_id": idx,
                "original_shape": image_np.shape,
                "num_classes": 2,
                "place": row["place"],  # Water / land
                "task_type": task_type,
            }
            sample = ImageSample(image_np, binary_label, metadata)
            samples.append(sample)

    class_descriptions = [
        "0: landbird",
        "1: waterbird",
    ]

    return samples, class_descriptions


# Update the registry type hints
IMAGE_DATASETS = {
    "mnist": load_mnist_data,
    "cifar10": load_cifar10_data,
    "fashion_mnist": load_fashion_mnist_data,
    "waterbird": load_waterbird_data,
}


def load_image_data(
    dataset_name: str, task_type: str = "classification"
) -> tuple[List[ImageSample], List[str]]:
    """Load any registered image dataset."""
    if dataset_name not in IMAGE_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(IMAGE_DATASETS.keys())}"
        )

    loader_func = IMAGE_DATASETS[dataset_name]
    return loader_func(task_type=task_type)
