#!/usr/bin/env python3
"""Protocol-locked metrics and splitting helpers for MMRDR Stage 22A."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


LESIONS = ("MA", "HE", "IH", "VB_IRMA", "NV", "VH", "RD")


def sample_id_hash(values: Iterable[object]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values)) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def fixed_split(
    frame: pd.DataFrame,
    validation_fraction: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use one immutable grade-stratified validation split for every run."""
    official_train = (
        frame.loc[frame["official_split"] == "train"]
        .copy()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    test = (
        frame.loc[frame["official_split"] == "test"]
        .copy()
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
    )
    train, validation = train_test_split(
        official_train,
        test_size=validation_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=official_train["grade"],
    )
    sets = [set(part["sample_id"].astype(str)) for part in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("train/validation/test sample overlap")
    return train, validation, test


def threshold_at_minimum_specificity(
    target: np.ndarray,
    score: np.ndarray,
    minimum_specificity: float = 0.90,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    if target.ndim != 1 or score.ndim != 1 or len(target) != len(score):
        raise ValueError("target and score must be same-length vectors")
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("binary target must contain both 0 and 1")
    if not np.isfinite(score).all():
        raise ValueError("score contains non-finite values")
    thresholds = np.r_[np.inf, np.unique(score)[::-1], -np.inf]
    feasible: list[tuple[float, float, float, int, int, int, int]] = []
    for threshold in thresholds:
        prediction = score >= threshold
        tp = int(np.sum((target == 1) & prediction))
        fn = int(np.sum((target == 1) & ~prediction))
        tn = int(np.sum((target == 0) & ~prediction))
        fp = int(np.sum((target == 0) & prediction))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        if specificity + 1e-12 >= minimum_specificity:
            feasible.append((sensitivity, specificity, float(threshold), tp, fn, tn, fp))
    if not feasible:
        raise RuntimeError("no threshold satisfies the specificity constraint")
    sensitivity, specificity, threshold, tp, fn, tn, fp = max(
        feasible, key=lambda item: (item[0], item[1], item[2])
    )
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def operating_metrics_at_threshold(
    target: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = score >= float(threshold)
    tp = int(np.sum((target == 1) & prediction))
    fn = int(np.sum((target == 1) & ~prediction))
    tn = int(np.sum((target == 0) & ~prediction))
    fp = int(np.sum((target == 0) & prediction))
    return {
        "threshold": float(threshold),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
    }


def expected_calibration_error(
    target: np.ndarray,
    score: np.ndarray,
    bins: int = 10,
) -> float:
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (score >= edges[index]) & (score <= edges[index + 1])
        else:
            mask = (score >= edges[index]) & (score < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(target[mask].mean()) - float(score[mask].mean()))
    return float(value if total else np.nan)


def screening_metrics_from_grade_probabilities(
    grade: np.ndarray,
    probability: np.ndarray,
    minimum_specificity: float = 0.90,
    fixed_threshold: float | None = None,
) -> dict:
    grade = np.asarray(grade, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if probability.shape != (len(grade), 5):
        raise ValueError(f"expected Nx5 grade probabilities, got {probability.shape}")
    if not np.isfinite(probability).all():
        raise ValueError("grade probabilities contain non-finite values")
    row_sum = probability.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-5):
        raise ValueError("grade probabilities do not sum to one")
    prediction = probability.argmax(axis=1)
    binary_matrix = label_binarize(grade, classes=np.arange(5))
    referable_target = (grade >= 2).astype(int)
    referable_score = probability[:, 2:].sum(axis=1)
    operating = (
        threshold_at_minimum_specificity(
            referable_target, referable_score, minimum_specificity
        )
        if fixed_threshold is None
        else operating_metrics_at_threshold(
            referable_target, referable_score, fixed_threshold
        )
    )
    precision, recall, per_f1, support = precision_recall_fscore_support(
        grade, prediction, labels=np.arange(5), zero_division=0
    )
    return {
        "samples": int(len(grade)),
        "accuracy": float(accuracy_score(grade, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(grade, prediction)),
        "macro_f1": float(f1_score(grade, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(grade, prediction, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(grade, prediction, weights="quadratic")),
        "grade_mae": float(mean_absolute_error(grade, prediction)),
        "macro_auc_ovr": float(roc_auc_score(binary_matrix, probability, average="macro", multi_class="ovr")),
        "macro_auprc": float(average_precision_score(binary_matrix, probability, average="macro")),
        "referable_prevalence": float(referable_target.mean()),
        "referable_auc": float(roc_auc_score(referable_target, referable_score)),
        "referable_auprc": float(average_precision_score(referable_target, referable_score)),
        "referable_brier": float(brier_score_loss(referable_target, referable_score)),
        "referable_ece_10bin": expected_calibration_error(referable_target, referable_score, bins=10),
        "sensitivity_at_specificity_0_90": float(operating["sensitivity"]),
        "specificity_at_selected_threshold": float(operating["specificity"]),
        "selected_threshold_specificity_0_90": float(operating["threshold"]),
        "threshold_source": "current_validation" if fixed_threshold is None else "locked_validation",
        "operating_confusion": {key: int(operating[key]) for key in ("tp", "fn", "tn", "fp")},
        "confusion_matrix": confusion_matrix(grade, prediction, labels=np.arange(5)).tolist(),
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index in range(5)
        },
    }


def lesion_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    targets = np.asarray(targets)
    masks = np.asarray(masks)
    probabilities = np.asarray(probabilities)
    aucs: list[float] = []
    auprcs: list[float] = []
    per_lesion: dict[str, dict] = {}
    for index, name in enumerate(LESIONS):
        mask = masks[:, index].astype(bool)
        target = targets[mask, index].astype(int)
        probability = probabilities[mask, index]
        item: dict[str, float | int] = {
            "labeled": int(mask.sum()),
            "positive": int((target == 1).sum()),
            "negative": int((target == 0).sum()),
            "prevalence_random_auprc": float(target.mean()) if len(target) else float("nan"),
        }
        if len(np.unique(target)) == 2:
            item["auc"] = float(roc_auc_score(target, probability))
            item["auprc"] = float(average_precision_score(target, probability))
            item["f1_fixed_0_5"] = float(f1_score(target, probability >= 0.5, zero_division=0))
            aucs.append(float(item["auc"]))
            auprcs.append(float(item["auprc"]))
        per_lesion[name] = item
    return {
        "lesion_macro_auc": float(np.mean(aucs)),
        "lesion_macro_auprc": float(np.mean(auprcs)),
        "per_lesion": per_lesion,
    }


def selection_key(metrics: dict) -> tuple[float, float, float]:
    """Primary UWF metric, then ordinal performance, then calibration."""
    return (
        float(metrics["sensitivity_at_specificity_0_90"]),
        float(metrics["quadratic_weighted_kappa"]),
        -float(metrics["referable_brier"]),
    )


def is_better_selection(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float] | None,
    primary_tolerance: float = 1e-12,
) -> bool:
    if incumbent is None:
        return True
    if candidate[0] > incumbent[0] + primary_tolerance:
        return True
    if abs(candidate[0] - incumbent[0]) <= primary_tolerance:
        if candidate[1] > incumbent[1] + 1e-12:
            return True
        if abs(candidate[1] - incumbent[1]) <= 1e-12 and candidate[2] > incumbent[2] + 1e-12:
            return True
    return False
