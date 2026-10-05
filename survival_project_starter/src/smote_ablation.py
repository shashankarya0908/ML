"""SMOTE-before-CV versus train-fold-only survival leakage ablation."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

from src.cv import (
    DEFAULT_SEEDS,
    N_SPLITS,
    c_index_from_log_hazard,
    stratified_fold_indices,
)
from src.data import EVENT_COLUMN, ID_COLUMN, TIME_COLUMN, feature_frame
from src.models import BASELINE_PENALIZER, fit_cox_model


RISK_LABEL_COLUMN = "smote_risk_label"
IS_REAL_COLUMN = "is_real_patient"
SMOTE_THRESHOLD_DAYS = 270.0


def prepare_smote_ablation_cohort(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    threshold_days: float = SMOTE_THRESHOLD_DAYS,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Drop unknown early censorings and create the binary SMOTE-only risk label."""
    censored_early = frame[EVENT_COLUMN].eq(0) & frame[TIME_COLUMN].lt(threshold_days)
    eligible = frame.loc[~censored_early].copy().reset_index(drop=True)
    high_risk = eligible[EVENT_COLUMN].eq(1) & eligible[TIME_COLUMN].lt(threshold_days)
    low_risk = eligible[TIME_COLUMN].ge(threshold_days)
    if not (high_risk | low_risk).all():
        raise ValueError("Every retained patient must receive a threshold-based SMOTE label.")
    labels = high_risk.astype(int).rename(RISK_LABEL_COLUMN)
    feature_frame(eligible, features)
    if labels.nunique() != 2:
        raise ValueError("The threshold-based SMOTE label must contain both classes.")
    return eligible, labels, int(censored_early.sum())


def smote_resample_survival_rows(
    data: pd.DataFrame,
    labels: pd.Series,
    features: tuple[str, ...],
    random_state: int,
) -> pd.DataFrame:
    """SMOTE scaled clinical/radiomic features plus time/event; restore survival units."""
    columns = (*features, TIME_COLUMN, EVENT_COLUMN)
    resample_matrix = data.loc[:, list(columns)].to_numpy(dtype=float)
    class_counts = labels.value_counts()
    minority_count = int(class_counts.min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two samples in each risk-label class.")
    neighbors = min(5, minority_count - 1)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(resample_matrix)
    sampler = SMOTE(k_neighbors=neighbors, random_state=random_state)
    resampled_standardized, resampled_labels = sampler.fit_resample(standardized, labels)
    original_count = len(data)
    if not np.array_equal(resampled_labels[:original_count], labels.to_numpy()):
        raise RuntimeError("SMOTE did not preserve original rows before synthetic rows.")
    if not np.allclose(resampled_standardized[:original_count], standardized):
        raise RuntimeError("SMOTE changed or reordered original rows.")

    restored = pd.DataFrame(
        scaler.inverse_transform(resampled_standardized),
        columns=columns,
    )
    restored[TIME_COLUMN] = restored[TIME_COLUMN].clip(lower=1.0)
    restored[EVENT_COLUMN] = restored[EVENT_COLUMN].round().clip(0, 1).astype(int)
    restored[RISK_LABEL_COLUMN] = np.asarray(resampled_labels, dtype=int)
    restored[IS_REAL_COLUMN] = np.arange(len(restored)) < original_count
    restored["smote_k_neighbors"] = neighbors
    restored[ID_COLUMN] = [
        *data[ID_COLUMN].tolist(),
        *(f"SMOTE-{random_state}-{index}" for index in range(len(restored) - original_count)),
    ]
    if not restored[TIME_COLUMN].ge(1).all():
        raise RuntimeError("SMOTE generated survival times below one day after clipping.")
    if not restored[EVENT_COLUMN].isin((0, 1)).all():
        raise RuntimeError("SMOTE generated event values outside {0, 1}.")
    return restored


def _fit_and_score(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[float, float]:
    """Fit fold-scaled CoxPH and score held-out survival observations."""
    fitted = fit_cox_model(
        feature_frame(training, features),
        training[TIME_COLUMN],
        training[EVENT_COLUMN],
        penalizer=BASELINE_PENALIZER,
    )
    log_hazard = fitted.predict_log_hazard(feature_frame(testing, features)).to_numpy()
    try:
        score = c_index_from_log_hazard(
            testing[TIME_COLUMN],
            log_hazard,
            testing[EVENT_COLUMN],
        )
    except (ZeroDivisionError, ValueError) as exc:
        raise RuntimeError("SMOTE ablation fold has an undefined C-index.") from exc
    return score, fitted.penalizer


def _evaluate_real_training_setups(
    data: pd.DataFrame,
    labels: pd.Series,
    features: tuple[str, ...],
    setup: str,
    smote_inside_fold: bool,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Evaluate D or C; the test fold always contains real patients only."""
    records: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold, (train_idx, test_idx) in enumerate(
            stratified_fold_indices(labels, N_SPLITS, seed),
            start=1,
        ):
            training = data.iloc[train_idx].reset_index(drop=True)
            train_labels = labels.iloc[train_idx].reset_index(drop=True)
            testing = data.iloc[test_idx].reset_index(drop=True)
            if smote_inside_fold:
                training = smote_resample_survival_rows(
                    training,
                    train_labels,
                    features,
                    random_state=seed * 100 + fold,
                )
            score, penalizer_used = _fit_and_score(training, testing, features)
            records.append(
                {
                    "setup": setup,
                    "seed": int(seed),
                    "fold": fold,
                    "n_train": len(training),
                    "n_train_real": int(
                        training[IS_REAL_COLUMN].sum()
                        if IS_REAL_COLUMN in training
                        else len(training)
                    ),
                    "n_train_synthetic": int(
                        (~training[IS_REAL_COLUMN]).sum()
                        if IS_REAL_COLUMN in training
                        else 0
                    ),
                    "n_test": len(testing),
                    "n_test_real": len(testing),
                    "n_test_synthetic": 0,
                    "smote_k_neighbors": int(
                        training["smote_k_neighbors"].iloc[0]
                        if "smote_k_neighbors" in training
                        else 0
                    ),
                    "penalizer_used": penalizer_used,
                    "c_index": score,
                }
            )
    return pd.DataFrame(records)


def _evaluate_pre_cv_smote_setups(
    data: pd.DataFrame,
    labels: pd.Series,
    features: tuple[str, ...],
    setup: str,
    score_synthetic_test_rows: bool,
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Evaluate B or A after intentionally SMOTE-augmenting before outer CV."""
    records: list[dict[str, float | int | str]] = []
    for seed in seeds:
        augmented = smote_resample_survival_rows(
            data,
            labels,
            features,
            random_state=seed,
        )
        augmented_labels = augmented[RISK_LABEL_COLUMN]
        for fold, (train_idx, test_idx) in enumerate(
            stratified_fold_indices(augmented_labels, N_SPLITS, seed),
            start=1,
        ):
            training = augmented.iloc[train_idx].reset_index(drop=True)
            augmented_test = augmented.iloc[test_idx].reset_index(drop=True)
            if score_synthetic_test_rows:
                testing = augmented_test
            else:
                testing = augmented_test.loc[augmented_test[IS_REAL_COLUMN]].copy()
            if testing.empty:
                raise RuntimeError(
                    f"{setup}, seed {seed}, fold {fold}: no real test observations to score."
                )
            score, penalizer_used = _fit_and_score(training, testing, features)
            records.append(
                {
                    "setup": setup,
                    "seed": int(seed),
                    "fold": fold,
                    "n_train": len(training),
                    "n_train_real": int(training[IS_REAL_COLUMN].sum()),
                    "n_train_synthetic": int((~training[IS_REAL_COLUMN]).sum()),
                    "n_test": len(augmented_test) if score_synthetic_test_rows else len(testing),
                    "n_test_real": int(augmented_test[IS_REAL_COLUMN].sum()),
                    "n_test_synthetic": int((~augmented_test[IS_REAL_COLUMN]).sum()),
                    "smote_k_neighbors": int(augmented["smote_k_neighbors"].iloc[0]),
                    "penalizer_used": penalizer_used,
                    "c_index": score,
                }
            )
    return pd.DataFrame(records)


def run_smote_ablation(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    threshold_days: float = SMOTE_THRESHOLD_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run D, C, B, A SMOTE variants with repeated event-label-stratified CV."""
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("The Task 7 ablation uses exactly five unique seeds.")
    data, labels, dropped = prepare_smote_ablation_cohort(
        frame,
        features,
        threshold_days,
    )
    setup_folds = (
        _evaluate_real_training_setups(
            data,
            labels,
            features,
            "D_no_smote_real_test",
            smote_inside_fold=False,
            seeds=seeds,
        ),
        _evaluate_real_training_setups(
            data,
            labels,
            features,
            "C_train_fold_smote_real_test",
            smote_inside_fold=True,
            seeds=seeds,
        ),
        _evaluate_pre_cv_smote_setups(
            data,
            labels,
            features,
            "B_pre_cv_smote_real_test",
            score_synthetic_test_rows=False,
            seeds=seeds,
        ),
        _evaluate_pre_cv_smote_setups(
            data,
            labels,
            features,
            "A_pre_cv_smote_all_test_rows",
            score_synthetic_test_rows=True,
            seeds=seeds,
        ),
    )
    fold_scores = pd.concat(setup_folds, ignore_index=True)
    for setup, setup_scores in fold_scores.groupby("setup"):
        assert len(setup_scores) == 10 * 5, f"{setup} must have exactly 50 folds."
        assert setup_scores.groupby("seed").size().eq(10).all()
        assert setup_scores["c_index"].notna().all()

    rows: list[dict[str, float | int | str]] = []
    for setup, setup_scores in fold_scores.groupby("setup", sort=False):
        repeat_means = setup_scores.groupby("seed")["c_index"].mean()
        rows.append(
            {
                "setup": setup,
                "n_real_patients": len(data),
                "n_dropped_censored_before_threshold": dropped,
                "n_high_risk": int(labels.sum()),
                "n_low_risk": int((labels == 0).sum()),
                "mean_fold_c_index": float(setup_scores["c_index"].mean()),
                "sd_fold_c_index": float(setup_scores["c_index"].std(ddof=1)),
                "mean_repeat_fold_c_index": float(repeat_means.mean()),
                "sd_repeat_fold_c_index": float(repeat_means.std(ddof=1)),
                "n_seeds": len(seeds),
                "n_folds_per_seed": N_SPLITS,
                "mean_test_rows": float(
                    setup_scores.groupby("seed")["n_test"].mean().mean()
                ),
            }
        )
    metadata = {
        "n_input_patients": len(frame),
        "n_eligible_patients": len(data),
        "n_dropped_censored_before_threshold": dropped,
        "n_high_risk": int(labels.sum()),
        "n_low_risk": int((labels == 0).sum()),
        "threshold_days": int(threshold_days),
    }
    return pd.DataFrame(rows), fold_scores, metadata
