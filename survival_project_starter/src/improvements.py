"""Training-fold-only survival model improvement experiments."""

from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.cv import (
    DEFAULT_SEEDS,
    N_SPLITS,
    c_index_from_log_hazard,
    stratified_fold_indices,
)
from src.data import EVENT_COLUMN, TIME_COLUMN, feature_frame
from src.models import BASELINE_PENALIZER, FittedCox, fit_cox_model


PENALIZER_GRID = (0.01, 0.1, 0.5, 1.0, 5.0, 10.0)
FitPredict: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame, int, int],
    tuple[np.ndarray, dict[str, float | int | str]],
]
FitDiagnostics: TypeAlias = dict[str, int]
SelectionCache: TypeAlias = dict[
    tuple[str, int, int, tuple[str, ...]],
    tuple[tuple[str, ...], FitDiagnostics],
]
DIAGNOSTIC_FIELDS = (
    "cox_fits",
    "non_converged_fits",
    "non_converged_attempts",
    "penalizer_raised_fits",
    "step_size_adjusted_fits",
    "unresolved_fits",
)


def _empty_diagnostics() -> FitDiagnostics:
    """Create an initialized convergence-accounting record."""
    return {field: 0 for field in DIAGNOSTIC_FIELDS}


def _record_fit(diagnostics: FitDiagnostics, fitted: FittedCox) -> None:
    """Count convergence retries and their handling for one Cox model."""
    diagnostics["cox_fits"] += 1
    diagnostics["non_converged_attempts"] += fitted.non_converged_attempts
    diagnostics["non_converged_fits"] += int(fitted.non_converged_attempts > 0)
    diagnostics["penalizer_raised_fits"] += int(fitted.penalizer_escalated)


def _evaluate_strategy(
    frame: pd.DataFrame,
    name: str,
    fit_predict: FitPredict,
    n_features: int,
    seeds: Sequence[int],
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Run a train-fold strategy through repeated, stratified outer CV."""
    if len(seeds) < 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least five unique, fixed random seeds.")
    events = frame[EVENT_COLUMN].reset_index(drop=True)
    times = frame[TIME_COLUMN].reset_index(drop=True)
    data = frame.reset_index(drop=True)
    fold_rows: list[dict[str, float | int | str]] = []
    repeat_means: list[float] = []
    pooled_scores: list[float] = []

    print(
        f"{name}: starting {len(seeds)} repeats of {N_SPLITS}-fold CV.",
        flush=True,
    )
    for seed in seeds:
        oof = np.full(len(data), np.nan, dtype=float)
        fold_values: list[float] = []
        for fold, (train_indices, test_indices) in enumerate(
            stratified_fold_indices(events, N_SPLITS, seed), start=1
        ):
            risk, details = fit_predict(
                data.iloc[train_indices].reset_index(drop=True),
                data.iloc[test_indices].reset_index(drop=True),
                int(seed),
                fold,
            )
            if len(risk) != len(test_indices) or not np.isfinite(risk).all():
                raise RuntimeError(f"{name}, seed {seed}, fold {fold}: invalid test risk scores.")
            try:
                fold_c = c_index_from_log_hazard(
                    times.iloc[test_indices],
                    risk,
                    events.iloc[test_indices],
                )
            except (ZeroDivisionError, ValueError) as exc:
                raise RuntimeError(
                    f"{name}, seed {seed}, fold {fold}: C-index is undefined."
                ) from exc
            oof[test_indices] = risk
            fold_values.append(fold_c)
            fold_rows.append(
                {
                    "model": name,
                    "seed": int(seed),
                    "fold": fold,
                    "n_train": len(train_indices),
                    "n_test": len(test_indices),
                    "test_deaths": int(events.iloc[test_indices].sum()),
                    "c_index": fold_c,
                    **details,
                }
            )
            if fold == 1 or fold % 5 == 0:
                print(
                    f"{name}: seed {seed}, completed outer folds "
                    f"{fold}/{N_SPLITS}.",
                    flush=True,
                )
        if not np.isfinite(oof).all():
            raise RuntimeError(f"{name}, seed {seed}: not every row received an OOF score.")
        repeat_means.append(float(np.mean(fold_values)))
        pooled_scores.append(c_index_from_log_hazard(times, oof, events))

    folds = pd.DataFrame(fold_rows)
    expected_folds = N_SPLITS * len(seeds)
    assert len(folds) == expected_folds, (
        f"{name} must have exactly {expected_folds} outer folds."
    )
    assert folds.groupby("seed").size().eq(N_SPLITS).all()
    assert folds["c_index"].notna().all()
    repeat_array = np.asarray(repeat_means, dtype=float)
    pooled_array = np.asarray(pooled_scores, dtype=float)
    summary: dict[str, float | int | str] = {
        "model": name,
        "n_patients": int(len(frame)),
        "n_features": int(n_features),
        "mean_fold_c_index": float(folds["c_index"].mean()),
        "sd_fold_c_index": float(folds["c_index"].std(ddof=1)),
        "mean_repeat_fold_c_index": float(repeat_array.mean()),
        "sd_repeat_fold_c_index": float(repeat_array.std(ddof=1)),
        "pooled_oof_c_index": float(pooled_array.mean()),
        "sd_pooled_oof_c_index": float(pooled_array.std(ddof=1)),
        "n_seeds": int(len(seeds)),
        "n_folds_per_seed": N_SPLITS,
    }
    for field in DIAGNOSTIC_FIELDS:
        summary[field] = int(folds[field].sum())
    return summary, folds


def _fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    penalizer: float = BASELINE_PENALIZER,
) -> tuple[np.ndarray, FittedCox]:
    """Fit one fold's Cox model with its own training-fitted scaler."""
    fitted = fit_cox_model(
        feature_frame(train, features),
        train[TIME_COLUMN],
        train[EVENT_COLUMN],
        penalizer=penalizer,
    )
    risk = fitted.predict_log_hazard(feature_frame(test, features)).to_numpy()
    return risk, fitted


def select_univariate_cox_features(
    train: pd.DataFrame,
    candidates: tuple[str, ...],
    k: int,
    diagnostics: FitDiagnostics | None = None,
) -> tuple[str, ...]:
    """Rank candidate features by univariate Cox p-value using training rows only."""
    if k < 1 or k > len(candidates):
        raise ValueError(f"k must be between 1 and {len(candidates)}.")
    ranked: list[tuple[float, float, str]] = []
    for feature in candidates:
        values = train[feature]
        if values.nunique() < 2:
            ranked.append((1.0, 0.0, feature))
            continue
        fitted = fit_cox_model(
            feature_frame(train, (feature,)),
            train[TIME_COLUMN],
            train[EVENT_COLUMN],
            penalizer=BASELINE_PENALIZER,
        )
        if diagnostics is not None:
            _record_fit(diagnostics, fitted)
        estimate = fitted.model.summary.loc[feature]
        p_value = float(estimate["p"])
        coefficient = float(estimate["coef"])
        ranked.append(
            (
                p_value if np.isfinite(p_value) else 1.0,
                -abs(coefficient) if np.isfinite(coefficient) else 0.0,
                feature,
            )
        )
    ranked.sort()
    return tuple(item[2] for item in ranked[:k])


def _cached_univariate_order(
    train: pd.DataFrame,
    candidates: tuple[str, ...],
    seed: int,
    fold: int,
    dataset_scope: str,
    cache: SelectionCache | None,
) -> tuple[tuple[str, ...], FitDiagnostics]:
    """Reuse one training-fold univariate ranking across related top-k models."""
    key = (dataset_scope, seed, fold, candidates)
    if cache is None:
        diagnostics = _empty_diagnostics()
        ranking = select_univariate_cox_features(
            train, candidates, len(candidates), diagnostics
        )
        return ranking, diagnostics
    if key not in cache:
        diagnostics = _empty_diagnostics()
        ranking = select_univariate_cox_features(
            train, candidates, len(candidates), diagnostics
        )
        cache[key] = (ranking, diagnostics)
    ranking, diagnostics = cache[key]
    return ranking, diagnostics.copy()


def _nested_best_penalizer(
    train: pd.DataFrame,
    features: tuple[str, ...],
    grid: tuple[float, ...],
    random_state: int,
) -> tuple[float, dict[float, float], FitDiagnostics]:
    """Select a penalty using only inner folds from the current outer training fold."""
    predictors = feature_frame(train, features).reset_index(drop=True)
    times = train[TIME_COLUMN].reset_index(drop=True)
    events = train[EVENT_COLUMN].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
    inner_scores: dict[float, list[float]] = {penalty: [] for penalty in grid}
    diagnostics = _empty_diagnostics()
    for inner_fold, (inner_train, inner_valid) in enumerate(
        splitter.split(predictors, events), start=1
    ):
        for penalty in grid:
            fitted = fit_cox_model(
                predictors.iloc[inner_train],
                times.iloc[inner_train],
                events.iloc[inner_train],
                penalizer=penalty,
            )
            _record_fit(diagnostics, fitted)
            risk = fitted.predict_log_hazard(predictors.iloc[inner_valid]).to_numpy()
            try:
                score = c_index_from_log_hazard(
                    times.iloc[inner_valid],
                    risk,
                    events.iloc[inner_valid],
                )
            except (ZeroDivisionError, ValueError) as exc:
                raise RuntimeError(
                    f"Nested penalty selection inner fold {inner_fold} has undefined C-index."
                ) from exc
            inner_scores[penalty].append(score)
    means = {penalty: float(np.mean(scores)) for penalty, scores in inner_scores.items()}
    best = max(grid, key=lambda penalty: (means[penalty], -grid.index(penalty)))
    return best, means, diagnostics


def nested_penalizer_grid_cv(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    grid: tuple[float, ...] = PENALIZER_GRID,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Evaluate inner-CV Cox penalizer selection across repeated outer folds."""
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("The nested penalizer grid must contain positive values.")

    def fit_predict(
        train: pd.DataFrame, test: pd.DataFrame, seed: int, fold: int
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        selected, _, diagnostics = _nested_best_penalizer(
            train,
            features,
            grid,
            random_state=seed * 100 + fold,
        )
        risk, fitted = _fit_and_predict(train, test, features, penalizer=selected)
        _record_fit(diagnostics, fitted)
        return risk, {
            "selected_penalizer": selected,
            "penalizer_used": fitted.penalizer,
            **diagnostics,
        }

    summary, folds = _evaluate_strategy(
        frame,
        "nested_penalizer_grid",
        fit_predict,
        len(features),
        seeds,
    )
    summary["mean_selected_penalizer"] = float(folds["selected_penalizer"].mean())
    return summary, folds


def univariate_cox_top_k_cv(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    k: int,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    selection_cache: SelectionCache | None = None,
    dataset_scope: str = "final_dataset",
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Evaluate top-k univariate Cox feature selection inside each outer fold."""

    def fit_predict(
        train: pd.DataFrame, test: pd.DataFrame, seed: int, fold: int
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        ranked, diagnostics = _cached_univariate_order(
            train,
            features,
            seed,
            fold,
            dataset_scope,
            selection_cache,
        )
        selected = ranked[:k]
        risk, fitted = _fit_and_predict(train, test, selected)
        _record_fit(diagnostics, fitted)
        return risk, {
            "selected_features": ";".join(selected),
            "penalizer_used": fitted.penalizer,
            **diagnostics,
        }

    return _evaluate_strategy(
        frame,
        f"univariate_cox_top_{k}",
        fit_predict,
        k,
        seeds,
    )


def radiomic_pca_cv(
    frame: pd.DataFrame,
    clinical_features: tuple[str, ...],
    radiomic_features: tuple[str, ...],
    n_components: int = 5,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Evaluate clinical features plus training-fold radiomic PCA components."""
    if n_components < 1 or n_components > min(len(radiomic_features), len(frame) - 1):
        raise ValueError("PCA component count must be valid for the radiomic cohort.")

    def fit_predict(
        train: pd.DataFrame, test: pd.DataFrame, seed: int, fold: int
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        train_radiomic = feature_frame(train, radiomic_features)
        test_radiomic = feature_frame(test, radiomic_features)
        radiomic_scaler = StandardScaler().fit(train_radiomic)
        train_scaled = radiomic_scaler.transform(train_radiomic)
        test_scaled = radiomic_scaler.transform(test_radiomic)
        pca = PCA(n_components=n_components, random_state=seed * 100 + fold)
        train_pcs = pca.fit_transform(train_scaled)
        test_pcs = pca.transform(test_scaled)
        component_names = tuple(f"radiomic_pc_{index + 1}" for index in range(n_components))
        train_features = pd.concat(
            [
                feature_frame(train, clinical_features).reset_index(drop=True),
                pd.DataFrame(train_pcs, columns=component_names),
            ],
            axis=1,
        )
        test_features = pd.concat(
            [
                feature_frame(test, clinical_features).reset_index(drop=True),
                pd.DataFrame(test_pcs, columns=component_names),
            ],
            axis=1,
        )
        fitted = fit_cox_model(
            train_features,
            train[TIME_COLUMN],
            train[EVENT_COLUMN],
        )
        diagnostics = _empty_diagnostics()
        _record_fit(diagnostics, fitted)
        risk = fitted.predict_log_hazard(test_features).to_numpy()
        return risk, {
            "pca_components": n_components,
            "radiomic_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "penalizer_used": fitted.penalizer,
            **diagnostics,
        }

    return _evaluate_strategy(
        frame,
        f"clinical_plus_radiomic_pca_{n_components}",
        fit_predict,
        len(clinical_features) + n_components,
        seeds,
    )


def clinical_plus_top_k_radiomics_cv(
    frame: pd.DataFrame,
    clinical_features: tuple[str, ...],
    radiomic_features: tuple[str, ...],
    k: int,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    selection_cache: SelectionCache | None = None,
    dataset_scope: str = "final_dataset",
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Evaluate clinical features plus top-k radiomics selected inside each fold."""

    def fit_predict(
        train: pd.DataFrame, test: pd.DataFrame, seed: int, fold: int
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        combined_order, diagnostics = _cached_univariate_order(
            train,
            clinical_features + radiomic_features,
            seed,
            fold,
            dataset_scope,
            selection_cache,
        )
        radiomic_order = tuple(
            feature for feature in combined_order if feature in radiomic_features
        )
        selected_radiomics = radiomic_order[:k]
        selected = clinical_features + selected_radiomics
        risk, fitted = _fit_and_predict(train, test, selected)
        _record_fit(diagnostics, fitted)
        return risk, {
            "selected_radiomic_features": ";".join(selected_radiomics),
            "penalizer_used": fitted.penalizer,
            **diagnostics,
        }

    return _evaluate_strategy(
        frame,
        f"clinical_plus_top_{k}_radiomics",
        fit_predict,
        len(clinical_features) + k,
        seeds,
    )


def stage_only_cv(
    frame: pd.DataFrame,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Evaluate the prespecified stage-only Cox baseline on the 52-patient cohort."""

    def fit_predict(
        train: pd.DataFrame, test: pd.DataFrame, _seed: int, _fold: int
    ) -> tuple[np.ndarray, dict[str, float | int | str]]:
        risk, fitted = _fit_and_predict(train, test, ("stage",))
        diagnostics = _empty_diagnostics()
        _record_fit(diagnostics, fitted)
        return risk, {"penalizer_used": fitted.penalizer, **diagnostics}

    return _evaluate_strategy(frame, "stage_only", fit_predict, 1, seeds)
