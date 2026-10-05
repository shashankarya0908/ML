"""Repeated, leakage-safe cross-validation for Cox models."""

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold

from src.data import EVENT_COLUMN, ID_COLUMN, TIME_COLUMN, feature_frame
from src.models import BASELINE_PENALIZER, fit_cox_model


DEFAULT_SEEDS = (17, 42, 73, 101, 2026)
N_SPLITS = 10


def stratified_fold_indices(
    labels: pd.Series | np.ndarray,
    n_splits: int,
    random_state: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield balanced stratified folds without sklearn's small-class warning."""
    values = np.asarray(labels)
    if values.ndim != 1 or pd.isna(values).any():
        raise ValueError("Stratification labels must be a one-dimensional, non-missing array.")
    if n_splits < 2 or len(values) < n_splits:
        raise ValueError("At least n_splits rows and two folds are required.")

    classes, first_indices, class_indices, class_counts = np.unique(
        values,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if len(classes) < 2:
        raise ValueError("Stratification requires at least two label classes.")

    if class_counts.min() >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        yield from splitter.split(np.zeros(len(values)), values)
        return

    rng = np.random.RandomState(random_state)
    class_order = np.argsort(first_indices)
    encoded_classes = np.empty_like(class_indices)
    for encoded_index, class_index in enumerate(class_order):
        encoded_classes[class_indices == class_index] = encoded_index
    ordered_classes = np.sort(encoded_classes)
    allocation = np.asarray(
        [
            np.bincount(ordered_classes[fold::n_splits], minlength=len(classes))
            for fold in range(n_splits)
        ]
    )
    fold_assignments = np.empty(len(values), dtype=int)
    for class_index in range(len(classes)):
        assignments = np.arange(n_splits).repeat(allocation[:, class_index])
        rng.shuffle(assignments)
        fold_assignments[encoded_classes == class_index] = assignments

    all_rows = np.arange(len(values))
    for fold_index in range(n_splits):
        test_indices = np.flatnonzero(fold_assignments == fold_index)
        train_mask = np.ones(len(values), dtype=bool)
        train_mask[test_indices] = False
        yield all_rows[train_mask], test_indices


def c_index_from_log_hazard(
    times: pd.Series | np.ndarray,
    log_hazard: pd.Series | np.ndarray,
    events: pd.Series | np.ndarray,
) -> float:
    """Compute survival concordance, reversing hazard because higher means shorter survival."""
    return float(concordance_index(times, -np.asarray(log_hazard), events))


def median_risk_groups(scores: pd.Series) -> tuple[pd.Series, float]:
    """Split log-hazard scores at the cohort median; ties at the median are high risk."""
    if scores.empty or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Risk scores must be non-empty and finite.")
    median = float(scores.median())
    groups = pd.Series(
        np.where(scores >= median, "high", "low"),
        index=scores.index,
        name="risk_group",
    )
    if groups.nunique() != 2:
        raise ValueError("Median risk split did not produce both groups.")
    return groups, median


def repeated_cox_cv(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    model_name: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    penalizer: float = BASELINE_PENALIZER,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate a Cox model with repeated stratified folds and train-only scaling."""
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least five unique, fixed random seeds.")
    predictors = feature_frame(frame, features)
    times = frame[TIME_COLUMN].reset_index(drop=True)
    events = frame[EVENT_COLUMN].reset_index(drop=True)
    ids = frame[ID_COLUMN].reset_index(drop=True)
    predictors = predictors.reset_index(drop=True)

    fold_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    repeat_fold_means: list[float] = []
    repeat_pooled_c: list[float] = []

    for seed in seeds:
        oof_log_hazard = np.full(len(frame), np.nan, dtype=float)
        seed_fold_scores: list[float] = []

        for fold, (train_indices, test_indices) in enumerate(
            stratified_fold_indices(events, N_SPLITS, seed), start=1
        ):
            fitted = fit_cox_model(
                predictors.iloc[train_indices],
                times.iloc[train_indices],
                events.iloc[train_indices],
                penalizer=penalizer,
            )
            risk = fitted.predict_log_hazard(predictors.iloc[test_indices]).to_numpy()
            try:
                fold_c = c_index_from_log_hazard(
                    times.iloc[test_indices],
                    risk,
                    events.iloc[test_indices],
                )
            except (ZeroDivisionError, ValueError) as exc:
                raise RuntimeError(
                    f"{model_name}, seed {seed}, fold {fold}: C-index is undefined."
                ) from exc
            oof_log_hazard[test_indices] = risk
            seed_fold_scores.append(fold_c)
            fold_rows.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "fold": fold,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "test_deaths": int(events.iloc[test_indices].sum()),
                    "penalizer_used": fitted.penalizer,
                    "c_index": fold_c,
                }
            )

        if not np.isfinite(oof_log_hazard).all():
            raise RuntimeError(f"{model_name}, seed {seed}: not every patient received an OOF score.")
        pooled_c = c_index_from_log_hazard(times, oof_log_hazard, events)
        repeat_fold_means.append(float(np.mean(seed_fold_scores)))
        repeat_pooled_c.append(pooled_c)
        prediction_rows.extend(
            {
                "model": model_name,
                "seed": int(seed),
                ID_COLUMN: ids.iloc[row],
                "oof_log_hazard": float(oof_log_hazard[row]),
            }
            for row in range(len(frame))
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    repeat_means = np.asarray(repeat_fold_means, dtype=float)
    pooled_scores = np.asarray(repeat_pooled_c, dtype=float)
    summary = {
        "model": model_name,
        "n_patients": int(len(frame)),
        "n_features": int(len(features)),
        "mean_fold_c_index": float(folds["c_index"].mean()),
        "sd_fold_c_index": float(folds["c_index"].std(ddof=1)),
        "mean_repeat_fold_c_index": float(repeat_means.mean()),
        "sd_repeat_fold_c_index": float(repeat_means.std(ddof=1)),
        "pooled_oof_c_index": float(pooled_scores.mean()),
        "sd_pooled_oof_c_index": float(pooled_scores.std(ddof=1)),
        "n_seeds": int(len(seeds)),
        "n_folds_per_seed": N_SPLITS,
    }
    return folds, predictions, summary
