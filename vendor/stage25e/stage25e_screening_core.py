#!/usr/bin/env python3
"""Locked Stage25E CFP operating point and model-selection helpers."""

from __future__ import annotations

import numpy as np

from stage22_screening_core import screening_metrics_from_grade_probabilities


def threshold_at_minimum_specificity_grade2_priority(
    grade: np.ndarray,
    score: np.ndarray,
    minimum_specificity: float = 0.90,
) -> dict[str, float | int]:
    """Select a finite threshold without viewing any test or external labels.

    Candidate order is locked as grade-2 sensitivity, overall referable-DR
    sensitivity, specificity, and then the higher threshold. Grade 2 is the
    prespecified difficult subgroup; grades >=2 define referable DR.
    """
    grade = np.asarray(grade, dtype=int)
    score = np.asarray(score, dtype=float)
    if grade.ndim != 1 or score.ndim != 1 or len(grade) != len(score):
        raise ValueError("grade and score must be same-length vectors")
    if not np.isfinite(score).all() or not ((0.0 <= score) & (score <= 1.0)).all():
        raise ValueError("score must contain finite probabilities in [0, 1]")
    if not set(np.unique(grade)).issubset(set(range(5))):
        raise ValueError("grade must contain only values 0 through 4")
    referable = grade >= 2
    grade2 = grade == 2
    nonreferable = grade < 2
    if not referable.any() or not nonreferable.any() or not grade2.any():
        raise ValueError("validation data require referable, nonreferable and grade-2 samples")
    reject_all = np.nextafter(float(score.max()), np.inf)
    thresholds = np.r_[reject_all, np.unique(score)[::-1]]
    feasible: list[tuple[float, float, float, float, int, int, int, int, int, int]] = []
    for threshold in thresholds:
        prediction = score >= threshold
        tp = int(np.sum(referable & prediction))
        fn = int(np.sum(referable & ~prediction))
        tn = int(np.sum(nonreferable & ~prediction))
        fp = int(np.sum(nonreferable & prediction))
        grade2_tp = int(np.sum(grade2 & prediction))
        grade2_fn = int(np.sum(grade2 & ~prediction))
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        grade2_sensitivity = grade2_tp / (grade2_tp + grade2_fn)
        if specificity + 1e-12 >= minimum_specificity:
            feasible.append(
                (
                    grade2_sensitivity,
                    sensitivity,
                    specificity,
                    float(threshold),
                    tp,
                    fn,
                    tn,
                    fp,
                    grade2_tp,
                    grade2_fn,
                )
            )
    if not feasible:
        raise RuntimeError("no threshold satisfies the specificity constraint")
    selected = max(feasible, key=lambda item: item[:4])
    grade2_sensitivity, sensitivity, specificity, threshold, tp, fn, tn, fp, grade2_tp, grade2_fn = selected
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "grade2_sensitivity": grade2_sensitivity,
        "grade2_tp": grade2_tp,
        "grade2_fn": grade2_fn,
        "grade2_samples": grade2_tp + grade2_fn,
    }


def cfp_screening_metrics(
    grade: np.ndarray,
    probability: np.ndarray,
    minimum_specificity: float = 0.90,
    fixed_threshold: float | None = None,
) -> dict:
    """Stage22 metrics with the Stage25E prespecified CFP operating point."""
    grade = np.asarray(grade, dtype=int)
    probability = np.asarray(probability, dtype=float)
    score = probability[:, 2:].sum(axis=1)
    if fixed_threshold is None:
        selected = threshold_at_minimum_specificity_grade2_priority(
            grade, score, minimum_specificity
        )
        threshold = float(selected["threshold"])
        source = "current_internal_validation_grade2_priority"
    else:
        threshold = float(fixed_threshold)
        prediction = score >= threshold
        grade2 = grade == 2
        referable = grade >= 2
        nonreferable = grade < 2
        selected = {
            "threshold": threshold,
            "sensitivity": float(prediction[referable].mean()),
            "specificity": float((~prediction[nonreferable]).mean()),
            "tp": int(np.sum(prediction & referable)),
            "fn": int(np.sum(~prediction & referable)),
            "tn": int(np.sum(~prediction & nonreferable)),
            "fp": int(np.sum(prediction & nonreferable)),
            "grade2_sensitivity": float(prediction[grade2].mean()),
            "grade2_tp": int(np.sum(prediction & grade2)),
            "grade2_fn": int(np.sum(~prediction & grade2)),
            "grade2_samples": int(grade2.sum()),
        }
        source = "locked_internal_validation_grade2_priority"
    metrics = screening_metrics_from_grade_probabilities(
        grade,
        probability,
        minimum_specificity=minimum_specificity,
        fixed_threshold=threshold,
    )
    metrics.update(
        {
            "sensitivity_at_specificity_0_90": float(selected["sensitivity"]),
            "specificity_at_selected_threshold": float(selected["specificity"]),
            "selected_threshold_specificity_0_90": threshold,
            "threshold_source": source,
            "operating_confusion": {
                key: int(selected[key]) for key in ("tp", "fn", "tn", "fp")
            },
            "grade2_sensitivity_at_selected_threshold": float(selected["grade2_sensitivity"]),
            "grade2_operating_confusion": {
                "tp": int(selected["grade2_tp"]),
                "fn": int(selected["grade2_fn"]),
            },
            "grade2_samples": int(selected["grade2_samples"]),
            "operating_point_priority": [
                "grade2_sensitivity",
                "referable_sensitivity",
                "specificity",
                "higher_threshold",
            ],
        }
    )
    return metrics


def cfp_checkpoint_selection_key(metrics: dict) -> tuple[float, float, float]:
    """Prespecified epoch key; exact ties retain the earlier epoch."""
    return (
        float(metrics["grade2_sensitivity_at_selected_threshold"]),
        float(metrics["sensitivity_at_specificity_0_90"]),
        float(metrics["referable_auc"]),
    )


def is_better_cfp_selection(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float] | None,
    tolerance: float = 1e-12,
) -> bool:
    if incumbent is None:
        return True
    for new, old in zip(candidate, incumbent):
        if new > old + tolerance:
            return True
        if new < old - tolerance:
            return False
    return False
