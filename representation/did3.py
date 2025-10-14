#!/usr/bin/env python3

"""
Domain-agnostic Dynamic ID3 implementation.
"""

import json
import logging
import random
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from typing import Any, Generator, Optional

import numpy as np
import wandb
from tqdm import tqdm

from domain import Domain
from feature_engine import Feature, check_feature_worker
from llm_generator import generate_features


logger = logging.getLogger(__name__)


class Node:
    def __init__(self, domain: Domain, parent: Optional["Internal"] = None, feature_candidates: list[Feature] = []):
        self._domain = domain
        self._parent = parent
        self._feature_candidates = feature_candidates

    def leaves(self) -> Generator["Leaf", None, None]:
        raise NotImplementedError

    def replace(self, before: "Node", after: "Node"):
        raise NotImplementedError

    def predict(self, x: Any) -> float:
        """
        Predict the label for a given domain input (x).
        The type of x is domain-specific; callers should pass domain.input_of(dp).
        """
        raise NotImplementedError

    @property
    def parent(self) -> Optional["Internal"]:
        return self._parent

    @property
    def error(self) -> float:
        """
        Mean absolute error of predictions on this node's own training examples.
        """
        raise NotImplementedError

    @cached_property
    def weight(self) -> int:
        """
        Number of examples this node contains.
        """
        raise NotImplementedError

    def feature_candidates_from_root(self) -> list[Feature]:
        return self._feature_candidates + ([] if not self._parent else self._parent.feature_candidates_from_root())


class Leaf(Node):
    def __init__(
        self, domain: Domain, examples: list[Any], parent: Optional["Internal"] = None, feature_candidates: list[Feature] = []
    ):
        super().__init__(domain, parent, feature_candidates)
        self._examples = examples

    def leaves(self) -> Generator["Leaf", None, None]:
        yield self

    def replace(self, before: "Node", after: "Node"):
        # Leaves have no children; nothing to replace.
        pass

    def predict(self, x: Any) -> float:
        # Use domain’s notion of leaf prediction (e.g., median label).
        return self._leaf_prediction

    @cached_property
    def _leaf_prediction(self) -> float:
        # Delegate statistic to the domain.
        return float(self._domain.leaf_prediction(self._examples))

    @cached_property
    def stats(self) -> dict:
        # Generic stats using domain labels
        labels = [self._domain.label_of(dp) for dp in self._examples]
        if not labels:
            return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
        return {
            "n": len(labels),
            "mean": float(np.mean(labels)),
            "median": float(np.median(labels)),
            "stdev": float(np.std(labels)),
        }

    @cached_property
    def error(self) -> float:
        # Delegate error computation to the domain
        return float(self._domain.leaf_error(self._examples))

    @cached_property
    def weight(self) -> int:
        return len(self._examples)


class Internal(Node):
    def __init__(
        self,
        domain: Domain,
        feature: Feature,
        threshold: float,
        left: Node,
        right: Node,
        parent: Optional["Internal"] = None,
        feature_candidates: list[Feature] = []
    ):
        super().__init__(domain, parent, feature_candidates)
        self._feature = feature
        self._threshold = threshold
        self._left = left
        self._left._parent = self
        self._right = right
        self._right._parent = self

    def leaves(self) -> Generator["Leaf", None, None]:
        yield from self._left.leaves()
        yield from self._right.leaves()

    def replace(self, before: "Node", after: "Node"):
        if self._left is before:
            self._left = after
            self._left._parent = self
        elif self._right is before:
            self._right = after
            self._right._parent = self

    def predict(self, x: Any) -> float:
        # Feature returns a scalar (or list -> take first), domain controls the input object shape.
        val = self._feature.execute(x)
        v = val[0] if isinstance(val, list) else val
        return (
            self._left.predict(x)
            if float(v) <= self._threshold
            else self._right.predict(x)
        )

    @property
    def error(self) -> float:
        # Weighted error of children
        n = self.weight
        return self._left.error * (self._left.weight / n) + self._right.error * (
            self._right.weight / n
        )

    @cached_property
    def weight(self) -> int:
        return self._left.weight + self._right.weight


class DynamicID3:
    """
    Domain-agnostic Dynamic ID3:
      - proposes new candidate features via an LLM prompt (domain formats the prompt),
      - finds best splits using domain.best_split_for_feature,
      - grows a decision tree while new splits reduce validation error.
    """

    def __init__(
        self,
        model: str,
        max_nodes: int = 1000,
        max_depth: int = 50,
        min_samples_split: int = 10,
        n_proposals: int = 10,
        n_examples: int = 10,
        min_side_ratio: float = 0.0,
    ):
        self._model = model
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._n_proposals = n_proposals
        self._min_side_ratio = min_side_ratio
        self._n_examples = n_examples

    def learn_features(
        self,
        domain,
        training_set: list[Any],
        validation_set: list[Any],
    ) -> list[str]:
        """
        Learn a representation by dynamically building a decision tree.
        Returns a list of feature codes (strings).
        """

        # Checkpoint file name: let the domain brand it if available.
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id = self._model.split('/')[-1]
        ckpt_name = f"did3_checkpoint_{ts}-{domain.domain_name()}__{model_id}.json"
        logger.info(f"Checkpoint file: {ckpt_name}")

        # Initialize tree with a single leaf node
        dt: Node = Leaf(domain, training_set)

        used_features: list[Feature] = []
        all_features: list[Feature] = []

        def error_on(node: Node, data: list[Any]) -> float:
            # Generic average error on a dataset: compare prediction vs. domain label
            if not data:
                return 0.0
            avg_err = 0.0
            for dp in data:
                x = domain.input_of(dp)
                y = domain.label_of(dp)
                pred = node.predict(x)
                avg_err += domain.prediction_error(pred, y) / len(data)
            return avg_err

        attempts = 0
        progress = True

        while attempts < self._max_nodes and progress:
            candidates = list(dt.leaves())
            candidates.sort(key=lambda n: n.error * n.weight, reverse=True)
            progress = False

            for node in candidates:
                if node.weight < self._min_samples_split:
                    continue
                attempts += 1

                # === 1) Propose new features using LLM ===
                proposals = self._propose_features(
                    domain=domain,
                    node=node,
                    n_examples=self._n_examples,
                    n_proposals=self._n_proposals,
                    feature_test_set=training_set[: min(10_000, len(training_set))],
                )
                if not proposals:
                    logger.warning(
                        "No valid feature proposals generated, skipping this node."
                    )
                    continue

                all_features.extend(proposals)

                candidates = proposals + node.feature_candidates_from_root()

                # === 2) Pick the best split among *all* discovered features so far on the path to the root ===
                # NOTE: We can use all features here instead; it works and typically gives better models,
                # but it is significantly more expensive especially in longer runs.
                split = self._find_best_split(
                    domain=domain,
                    features=candidates,
                    node=node,
                    min_side_ratio=self._min_side_ratio,
                )

                feature, threshold, left_data, right_data = split
                if not feature:
                    logger.info("No good split for this candidate.")
                    continue

                # Grow the tree
                left, right = Leaf(domain, left_data), Leaf(domain, right_data)
                new_node = Internal(
                    domain,
                    feature,
                    threshold,
                    left=left,
                    right=right,
                    parent=node.parent,
                    feature_candidates=proposals,
                )

                logger.info(f"Stats before split: {node.stats}")
                logger.info(f"Stats left: {left.stats}, right: {right.stats}")
                used_features.append(feature)

                # Splice into the tree
                if node.parent is None:
                    dt = new_node
                else:
                    node.parent.replace(node, new_node)

                error_val = error_on(dt, validation_set)
                error_delta = error_val - error_on(node, validation_set)
                wandb.log({"error": error_val, "error_delta": error_delta})

                logger.info(
                    f"Error on validation set after split: {error_val:.4f} (delta {error_delta:+.4f})"
                )

                try:
                    with open(ckpt_name, "w") as f:
                        json.dump(
                            {
                                "used_features": [f.code for f in used_features],
                                "all_features": [f.code for f in all_features],
                            },
                            f,
                        )
                except Exception as e:
                    logger.warning(f"Failed to write checkpoint {ckpt_name}: {e}")

                progress = True
                break

        return [f.code for f in used_features]

    def _find_best_split(
        self,
        domain: Domain,
        features: list[Feature],
        node: Node,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[Any], list[Any]]:
        """
        Iterate over candidate features and let the Domain compute the best split
        for each, then choose the split with the lowest error.
        """
        best = (
            None,
            float("inf"),
            [],
            [],
        )  # type: tuple[Optional[Feature], float, list[Any], list[Any]]
        best_err = float("inf")
        examples = getattr(node, "_examples", None)
        if examples is None:
            return best

        for feat in tqdm(features, f'Attempting to split on each existing feature ({len(examples)} examples in this leaf)...'):
            # Use domain's specialized splitter (e.g., RunningMedianAbs in chess).
            feat_, thr, left, right, err = domain.best_split_for_feature(
                examples, feat, min_side_ratio
            )
            if feat_ is not None and err < best_err:
                best = (feat_, thr, left, right)
                best_err = err

        return best

    def _propose_features(
        self,
        domain: Domain,
        node: Leaf,
        n_examples: int,
        n_proposals: int,
        feature_test_set: list[Any],
        timeout_s: float = 10,
    ) -> list[Feature]:
        """
        Call LLM to produce k candidate features for splitting the given leaf.
        """

        split_context = self._format_subtree_path(node)
        example_slice = random.sample(
            node._examples, min(n_examples, len(node._examples))
        )

        prompt = domain.format_split_prompt(
            n_output_features=n_proposals,
            examples=example_slice,
            split_context=split_context,
        )

        proposals: list[str] = generate_features(self._model, prompt)
        validation_input_data = [domain.input_of(x) for x in feature_test_set]
        tested_features = []

        for code in proposals:
            # Test feature on all positions in the "test set" to make sure it doesn't crash.
            try:
                ctx = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                    fut = ex.submit(check_feature_worker, code,
                                    validation_input_data, domain)
                    fut.result(timeout=timeout_s)
                    feature = Feature(code, domain)
                    tested_features.append(feature)
            except Exception as e:
                print(f"Error executing feature:\n{code}: {e}")

        for f in tested_features:
            logger.info(f"Working feature:\n{f.code}")

        return tested_features

    def _format_subtree_path(self, node: Leaf) -> str:
        """
        Format path from root to the leaf.
        """
        subtree_path = []

        while node.parent is not None:
            sign = "<" if node.parent._left is node else ">"
            subtree_path.append(
                f'value {sign} {node.parent._threshold:.3f} for "{node.parent._feature.description}" '
            )
            node = node.parent

        subtree_path.append("[root]\n")
        return " -> ".join(reversed(subtree_path))
