#!/usr/bin/env python3

from typing import Any
import json
import logging
import os

import wandb
import torch
import torch.nn
import chess
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from .trainer import Trainer
from .utils import evaluate_regression_model, prepare_train_valid_split
from chess_position import ChessPosition
from value import RFValueFunction


logger = logging.getLogger(__name__)


BOS = 0
EOS = 1
SEP = 2
PAD = 3
MAX_LENGTH = 100


def win_probability_to_token_index(
        wp: float,  # Between 0 and 100
        n_buckets: int,
        first_token_index: int,
) -> int:
    return min(first_token_index + round((wp / 100) * n_buckets),
               first_token_index + n_buckets - 1)


def _token_prefix_for_fen(fen: str) -> list[int]:
    pos_tokens = [ord(c) for c in fen]
    return [BOS] + pos_tokens + [PAD] * (MAX_LENGTH - 4 - len(pos_tokens)) + [SEP]


class ChessTransformer(torch.nn.Module):
    def __init__(self, bins: int, config: LlamaConfig):
        super().__init__()
        self.n_bins = bins
        self.model = LlamaForCausalLM(config)
        self.parallel = False

        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print('DataParallel on', n_gpus, 'devices.')
            self.model.to('cuda:0')
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(n_gpus)))
            self.parallel = True


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, positions: list[chess.Board]) -> list[float]:
        BATCH_SIZE = 64

        self.eval()
        predictions = []
        bin_weights = 100 * torch.linspace(0, 1, self.n_bins, device=self.model.device)

        with torch.no_grad():
            for i in range(0, len(positions), BATCH_SIZE):
                batch = positions[i:i+BATCH_SIZE]

                X = torch.tensor(
                    [_token_prefix_for_fen(pos.fen()) for pos in batch],
                    device=self.model.device,
                    dtype=torch.long,
                )
                outputs = self.model(X)
                logits = outputs.logits
                wp_logits = logits[:, -2, 128:(128 + self.n_bins)]
                wp_preds = (wp_logits.softmax(dim=-1) * bin_weights).sum(dim=-1)
                predictions.extend(wp_preds.cpu().tolist())

        return predictions

    def __call__(self, positions: list[chess.Board]) -> list[float]:
        # For this value function interface, behavelike a value function.
        evaluations = self.predict(positions)
        # Multiply by -1 if black is playing.
        return [e * (1 if pos.turn == chess.WHITE else -1)
                for e, pos in zip(evaluations, positions)]


class TransformerTrainer(Trainer):
    """Trains a Transformer baseline based on the Amortized Search paper.

    JAX implementation reference:
    https://github.com/google-deepmind/searchless_chess/blob/main/src/transformer.py
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        lr: float,
        batch_size: int,
        n_steps: int,
        n_output_bins: int = 128,
    ):
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_output_bins = n_output_bins
        self.lr = lr
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Vocab: character-level + output bins
        vocab_size = 128 + n_output_bins

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=embedding_dim,
            intermediate_size=4 * embedding_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            max_position_embeddings=MAX_LENGTH,
        )

        self.model = ChessTransformer(n_output_bins, config)
        if not self.model.parallel:
            self.model.to(self.device)

        model_size = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Initialized Transformer model with {model_size // 10**6}M parameters.")

    def _positions_to_tokens(self, positions: list[ChessPosition]) -> list[list[int]]:
        sequences = []
        for pos in positions:
            fen = pos.fen

            wp_token = win_probability_to_token_index(
                pos.evaluation,
                self.n_output_bins,
                first_token_index=128
            )
            s = _token_prefix_for_fen(fen) + [wp_token, EOS]
            assert len(s) == MAX_LENGTH, "Tokenization went over max model's length."
            sequences.append(s)
        return torch.tensor(sequences, device=self.device)

    def train(
        self,
        train_positions: list[ChessPosition],
        val_positions: list[ChessPosition],
        _eval_positions: list[ChessPosition] = None,
    ):
        CHECKPOINT_STEPS = [10**2, 10**3, 10**4, 25*10**3, 5*10**4, 10**5, 10**5 + 5*10**4, 2*10**5]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        self.model.train()
        last_loss = None
        initial_step = 0

        # Only compute loss for the next token after SEP.
        wp_token_mask = torch.zeros((self.batch_size, MAX_LENGTH), dtype=torch.int64, device=self.device)
        wp_token_mask[:, MAX_LENGTH - 2] = 1

        def checkpoint(step: int, path: str):
            torch.save({'steps': step, 'loss': last_loss, 'model': self.model, 'optimizer': optimizer.state_dict()}, path)
            print('Saved', path)

        def load_checkpoint_if_exists(path: str):
            if not os.path.exists(path):
                print('No checkpoint at', path)
                return

            ckpt = torch.load(path, weights_only=False)
            self.model = ckpt['model']
            nonlocal initial_step
            initial_step = ckpt['steps']
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Loaded from', path, 'at step', initial_step)

        load_checkpoint_if_exists('results/transformer_ckpt_last.pt')

        for step in tqdm(range(initial_step, self.n_steps)):
            batch_indices = torch.randint(0, len(train_positions), (self.batch_size,))
            batch = self._positions_to_tokens([train_positions[i] for i in batch_indices])
            y = batch*wp_token_mask + (1 - wp_token_mask) * -100
            outputs = self.model.forward(batch, labels=y)
            loss = outputs.loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
            wandb.log({"train/loss": last_loss, "step": step})

            if step in CHECKPOINT_STEPS:
                checkpoint(step, f'results/transformer_ckpt_{step}.pt')
            if step and step % 10000 == 0:
                checkpoint(step, f'results/transformer_ckpt_last.pt')

        checkpoint('results/chess_transformer.pt')

        return self.model, {'train_loss': last_loss}
