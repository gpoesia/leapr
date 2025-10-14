#!/usr/bin/env python3

import logging
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import wandb
import openai

from .trainer import Trainer

logger = logging.getLogger(__name__)


class ZeroShotLLMTrainer(Trainer):
    def __init__(
        self,
        model: str = "gpt-5-nano",
        reasoning_effort: str = "low",
        text_verbosity: str = "low",
        **kwargs,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.client = openai.OpenAI()
        self.last_request_time = 0
        self.min_delay = 1.0  # Rate limiting: 1 second between requests

    def _classify(self, text: str) -> int:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

        for attempt in range(2):  # Retry once on failure
            try:
                response = self.client.responses.create(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    instructions="You are a text classifier. Reply with only a single digit: 0 for human-written text, 1 for AI-generated text.",
                    input=f"Classify this text:\n\n{text[:1000]}",
                    text={"verbosity": self.text_verbosity},
                )
                result = response.output_text.strip()
                self.last_request_time = time.time()
                return int(result) if result.isdigit() else 0

            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt == 0:  # Only sleep on first failure
                    time.sleep(2)

        return 0  # Fallback after all retries

    def train(self, train_samples, val_samples, eval_samples=None):
        logger.info(f"Starting zero-shot classification with {self.model}")

        logger.info(f"Classifying {len(val_samples)} validation samples...")
        val_preds = [self._classify(s.text) for s in val_samples]
        val_acc = accuracy_score([s.target for s in val_samples], val_preds)
        logger.info(f"Validation accuracy: {val_acc:.4f}")

        eval_acc = None
        if eval_samples:
            logger.info(f"Classifying {len(eval_samples)} evaluation samples...")
            eval_preds = [self._classify(s.text) for s in eval_samples]
            eval_acc = accuracy_score([s.target for s in eval_samples], eval_preds)
            logger.info(f"Evaluation accuracy: {eval_acc:.4f}")

        wandb.log({"val_accuracy": val_acc, "eval_accuracy": eval_acc})

        return {}, {
            "train": {"accuracy": 0.0},
            "valid": {"accuracy": val_acc},
            "eval": {"accuracy": eval_acc} if eval_acc else None,
        }


class EmbeddingTrainer(Trainer):
    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 50,
        **kwargs,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.client = openai.OpenAI()

    def _embed_batch(self, texts):
        """Get embeddings with simple batch processing and error handling."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(
                f"Getting embeddings for batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}"
            )

            for attempt in range(2):
                try:
                    response = self.client.embeddings.create(
                        model=self.embedding_model, input=batch
                    )
                    batch_embeddings = [d.embedding for d in response.data]
                    all_embeddings.extend(batch_embeddings)
                    time.sleep(0.2)  # Small delay between batches
                    break

                except Exception as e:
                    logger.warning(f"Embedding batch attempt {attempt + 1} failed: {e}")
                    if attempt == 0:
                        time.sleep(2)
                    else:
                        # Fallback: create zero embeddings for failed batch
                        logger.error(f"Batch failed completely, using zero embeddings")
                        zero_embedding = [0.0] * 1536  # Default embedding dimension
                        all_embeddings.extend([zero_embedding] * len(batch))

        return np.array(all_embeddings)

    def train(self, train_samples, val_samples, eval_samples=None):
        logger.info(f"Training with {self.embedding_model} embeddings")

        train_texts = [s.text[:8000] for s in train_samples]  # Truncate for API limits
        train_labels = [s.target for s in train_samples]

        logger.info("Getting training embeddings...")
        X_train = self._embed_batch(train_texts)

        logger.info("Getting validation embeddings...")
        X_val = self._embed_batch([s.text[:8000] for s in val_samples])

        logger.info("Training classifier...")
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, train_labels)

        val_acc = clf.score(X_val, [s.target for s in val_samples])
        logger.info(f"Validation accuracy: {val_acc:.4f}")

        eval_acc = None
        if eval_samples:
            logger.info("Getting evaluation embeddings...")
            X_eval = self._embed_batch([s.text[:8000] for s in eval_samples])
            eval_acc = clf.score(X_eval, [s.target for s in eval_samples])
            logger.info(f"Evaluation accuracy: {eval_acc:.4f}")

        wandb.log(
            {
                "val_accuracy": val_acc,
                "eval_accuracy": eval_acc,
            }
        )

        return clf, {
            "valid": {"accuracy": val_acc},
            "eval": {"accuracy": eval_acc} if eval_acc else None,
        }
