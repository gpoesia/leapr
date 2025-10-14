#!/usr/bin/env python3

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

import wandb
import numpy as np
from sklearn.metrics import accuracy_score

from .trainer import Trainer

logger = logging.getLogger(__name__)


class NeuralNetworkBaselineTrainer(Trainer):
    """Neural network baseline trainer for image classification."""

    def __init__(
        self,
        model_name: str,
        initialization: str = "imagenet",  # "imagenet" or "random"
        lr: float = 1e-3,
        batch_size: int = 32,
        n_steps: int = 1000,
        num_classes: int = 10,
        **kwargs,
    ):
        self.model_name = model_name.lower()
        self.initialization = initialization
        self.lr = lr
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        """Create model based on config."""
        weights = "DEFAULT" if self.initialization == "imagenet" else None

        if self.model_name == "resnet50":
            model = models.resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif self.model_name == "efficientnetv2":
            model = models.efficientnet_v2_s(weights=weights)
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, self.num_classes
            )
        elif self.model_name == "squeezenet":
            model = models.squeezenet1_1(weights=weights)
            model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=1)
            model.num_classes = self.num_classes
        elif self.model_name == "vit":
            model = models.vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, self.num_classes)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model.to(self.device)

    # FIXME: input and output type annotation later
    # image_samples: list[ImageSample]
    def _prepare_data(self, image_samples):
        """Convert ImageSample objects to tensors."""
        images = []
        labels = []

        for sample in image_samples:
            img = sample.image  # Already np.ndarray from ImageSample

            # Handle different image formats
            if len(img.shape) == 2:  # Grayscale (MNIST: 28x28)
                img = np.stack([img] * 3, axis=0)  # Convert to 3x28x28
            elif len(img.shape) == 3:  # CIFAR-10: 3x32x32 or 32x32x3
                if img.shape[0] == 3:  # Already in CHW format
                    pass
                elif img.shape[2] == 3:  # HWC format, convert to CHW
                    img = img.transpose(2, 0, 1)
                else:  # Single channel in HWC, convert to CHW and replicate
                    img = np.stack([img.squeeze()] * 3, axis=0)

            # Normalize to [0,1]
            img = img.astype(np.float32) / 255.0

            # Resize for ViT if needed (ViT expects 224x224)
            target_size = 224
            if self.model_name == "vit" and img.shape[1] != target_size:
                import torch.nn.functional as F

                img_tensor = torch.tensor(img).unsqueeze(0)  # Add batch dim
                img_tensor = F.interpolate(
                    img_tensor,
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                img = img_tensor.squeeze(0).numpy()  # Remove batch dim

            images.append(img)
            labels.append(sample.target)

        return torch.tensor(np.array(images)), torch.tensor(labels, dtype=torch.long)

    # FIXME: input and output type annotation later
    # train_samples: list[ImageSample]
    def train(self, train_samples, val_samples, eval_samples=None):
        logger.info(
            f"Training {self.model_name} with {self.initialization} initialization"
        )

        # FIXME: change this later
        if train_samples and hasattr(train_samples[0], "metadata"):
            self.num_classes = train_samples[0].metadata.get(
                "num_classes", self.num_classes
            )

        model = self._create_model()

        # Prepare data from ImageSample objects
        X_train, y_train = self._prepare_data(train_samples)
        X_val, y_val = self._prepare_data(val_samples)

        # Prepare eval data if available
        X_eval, y_eval = None, None
        if eval_samples:
            X_eval, y_eval = self._prepare_data(eval_samples)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        step = 0

        # Log ~10 times total
        log_every = max(10, self.n_steps // 10)

        logger.info(f"Starting training for {self.n_steps} steps...")

        while step < self.n_steps:
            for batch_x, batch_y in train_loader:
                if step >= self.n_steps:
                    break

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                if step % log_every == 0:
                    logger.info(f"Step {step}/{self.n_steps}, Loss: {loss.item():.4f}")

                    # Log training accuracy periodically
                    model.eval()
                    with torch.no_grad():
                        val_acc = self._evaluate(model, X_val, y_val)

                        # Add eval accuracy if available
                        eval_acc = None
                        if X_eval is not None:
                            eval_acc = self._evaluate(model, X_eval, y_eval)
                    model.train()

                    log_dict = {
                        "train/loss": loss.item(),
                        "val/accuracy": val_acc,
                        "step": step,
                    }

                    if eval_acc is not None:
                        log_dict["eval/accuracy"] = eval_acc
                        logger.info(f"Val: {val_acc:.3f}, Eval: {eval_acc:.3f}")
                    else:
                        logger.info(f"Val: {val_acc:.3f}")

                    wandb.log(log_dict)

                step += 1

        logger.info(f"Training completed after {step} steps")

        # Evaluate
        logger.info("Evaluating model performance...")
        model.eval()
        with torch.no_grad():
            val_acc = self._evaluate(model, X_val, y_val)

        wandb.log({"final_val_accuracy": val_acc})

        logger.info(f"Val accuracy: {val_acc:.4f}")

        metrics = {
            "val": {"accuracy": val_acc},
            "eval": None,
        }

        if eval_samples:
            eval_acc = self._evaluate(model, X_eval, y_eval)
            metrics["eval"] = {"accuracy": eval_acc}
            logger.info(f"Eval accuracy: {eval_acc:.4f}")
            wandb.log({"final_eval_accuracy": eval_acc})

        return model, metrics

    def _evaluate(self, model, X, y):
        """Evaluate model accuracy."""
        model.eval()
        all_preds = []

        # Use larger batch size for evaluation
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu()
                all_preds.extend(preds.tolist())

        return accuracy_score(y.cpu(), all_preds)
