#!/usr/bin/env python3

"""
Implementation of the Text Classification Domain abstraction.
"""

from typing import Any, Optional, List
import math
import random
from collections import Counter

from trainer.random_forest import RandomForestTrainer

from . import Domain
from text_sample import TextSample, load_text_data
from feature_engine import Feature
from prompt_builder import load_prompt_template, format_text_api_description

DataPoint = TextSample


class TextClassification(Domain):
    def __init__(self):
        self._split_prompt_template = load_prompt_template(
            "prompts/text_classification_split.txt"
        )
        self._funsearch_prompt_template = load_prompt_template(
            "prompts/text_classification_funsearch.txt"
        )
        self.class_descriptions = None

    def domain_name(self) -> str:
        return "text_classification"

    def set_class_descriptions(self, descriptions: List[str]):
        """Store class descriptions for later use in prompts"""
        self.class_descriptions = descriptions

    def load_dataset(self, path: str, max_size: int) -> list[DataPoint]:
        # load_text_data returns a tuple (texts, class_descriptions)
        return load_text_data(path, task_type="classification")[0][:max_size]

    def format_split_prompt(
        self,
        n_output_features: int,
        examples: list[Any],
        split_context: Optional[str],
    ) -> str:
        api = format_text_api_description()

        class_descriptions_text = ""
        if self.class_descriptions:
            classes_text = ", ".join(self.class_descriptions)
            class_descriptions_text = f"This is a classification task with the following classes: {classes_text}."

        def format_sample_basic(sample: TextSample) -> str:
            text_preview = (
                sample.text[:200] + "..." if len(sample.text) > 200 else sample.text
            )
            return f"Sample: '{text_preview}', (Target: {sample.target})"

        examples_str = "\n".join([format_sample_basic(ex) for ex in examples])

        return self._split_prompt_template.format(
            class_descriptions=class_descriptions_text,
            api_description=api,
            subtree_path=split_context,
            examples=examples_str,
            num_features=n_output_features,
        )

    def format_funsearch_prompt(
        self,
        n_output_features: int,
        existing_features_with_importances: list[tuple[Feature, float]],
    ) -> str:
        api = format_text_api_description()

        def format_features_with_importances(f: Feature, importance: float) -> str:
            return f"Feature:\n{f.code}\nImportance: {importance:.3f}\n---\n"

        features_str = (
            "\n\n".join(
                [
                    format_features_with_importances(f, imp)
                    for f, imp in existing_features_with_importances
                ]
            )
            if existing_features_with_importances
            else "<No features yet>"
        )

        return self._funsearch_prompt_template.format(
            api_description=api,
            num_features=n_output_features,
            features=features_str,
        )

    def leaf_prediction(self, datapoints: list[DataPoint]) -> float:
        if not datapoints:
            return 0.0

        # Return most common class
        targets = [dp.target for dp in datapoints]
        return float(max(set(targets), key=targets.count))

    def leaf_error(self, datapoints: list[DataPoint]) -> float:
        n = len(datapoints)
        if n <= 1:
            return 0.0

        pred = self.leaf_prediction(datapoints)
        # Classification error rate (0.0 to 1.0)
        return sum(1 for dp in datapoints if dp.target != pred) / n

    def code_execution_namespace(self) -> dict[str, Any]:
        import re

        return {
            "math": math,
            "random": random,
            "re": re,
            "len": len,
            "str": str,
            "TextSample": TextSample,
        }

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        try:
            rows = []
            for sample in examples:
                vals = feature.execute(sample.text)
                v = vals[0] if isinstance(vals, list) else vals
                rows.append((float(v), sample.target, sample))
        except Exception:
            return None, math.inf, [], [], math.inf

        if not rows:
            return None, math.inf, [], [], math.inf

        # Sort by feature value
        rows.sort(key=lambda t: t[0])
        feats = [t[0] for t in rows]
        labels = [t[1] for t in rows]
        samples = [t[2] for t in rows]
        n = len(rows)
        if n <= 1:
            return None, math.inf, [], [], math.inf

        # Initialize left/right histograms
        left_cnt: Counter = Counter()
        right_cnt: Counter = Counter(labels)

        best_obj = math.inf  # weighted entropy
        best_idx = -1

        # Sweep possible split points between i and i+1
        for i in range(n - 1):
            # move item i from right -> left
            lbl = labels[i]
            left_cnt[lbl] += 1
            right_cnt[lbl] -= 1
            if right_cnt[lbl] == 0:
                del right_cnt[lbl]

            # only consider split if feature value changes at boundary
            if feats[i] == feats[i + 1]:
                continue

            n_left = i + 1
            n_right = n - n_left

            if min(n_left, n_right) < min_side_ratio * n:
                continue

            h_left = _histogram_entropy(left_cnt)
            h_right = _histogram_entropy(right_cnt)
            obj = (h_left * n_left + h_right * n_right) / n

            # Change best sweep if this one has lower entropy
            if obj < best_obj:
                best_obj = obj
                best_idx = i

        if best_idx == -1:
            return None, math.inf, [], [], math.inf

        threshold = feats[best_idx]  # left: <= threshold, right: > threshold
        left_samples = samples[: best_idx + 1]
        right_samples = samples[best_idx + 1 :]

        return feature, threshold, left_samples, right_samples, best_obj

    def input_of(self, dp: DataPoint) -> Any:
        return dp.text

    def label_of(self, dp: DataPoint) -> float:
        return dp.target

    def prediction_error(self, pred: Any, label: Any) -> float:
        return float(pred != label)  # 0.0 if correct, 1.0 if wrong

    def train_and_evaluate_simple_predictor(
        self,
        all_features: list[Feature],
        training_set: list[DataPoint],
        validation_set: list[DataPoint],
        training_parameters: dict[str, Any] = {},
    ) -> tuple[Any, float, float]:

        trainer = RandomForestTrainer(
            features_spec={"features": [f.code for f in all_features]},
            task_type="classification",
            domain_name=self.domain_name(),
            model_type="base_predictor",
            **training_parameters,
        )

        model, metrics = trainer.train(training_set, validation_set, None)

        # FIXME: we'll have to explain the use of F1 on the paper, should be easy
        return model, 1 - metrics["train"]["f1"], 1 - metrics["valid"]["f1"]


def _histogram_entropy(cnt: Counter) -> float:
    """Shannon entropy (log base 2) of a label-count histogram."""
    total = sum(cnt.values())
    if total <= 1:
        return 0.0
    h = 0.0
    for c in cnt.values():
        if c == 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h
