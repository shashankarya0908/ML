"""Run the first-stage repeated-seed Cox cross-validation analysis."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.cv import DEFAULT_SEEDS, median_risk_groups, repeated_cox_cv
from src.data import (
    EVENT_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
    PROJECT_ROOT,
    feature_frame,
    load_cohorts,
)
from src.improvements import (
    PENALIZER_GRID,
    SelectionCache,
    clinical_plus_top_k_radiomics_cv,
    nested_penalizer_grid_cv,
    radiomic_pca_cv,
    stage_only_cv,
    univariate_cox_top_k_cv,
)
from src.models import BASELINE_PENALIZER, fit_cox_model
from src.smote_ablation import run_smote_ablation
from src.plots import (
    plot_kaplan_meier_comparison,
    plot_mean_covariate_survival,
    plot_smote_ablation,
    plot_top_coefficients,
)


RESULTS_DIR = PROJECT_ROOT / "results"


def run_task_1() -> pd.DataFrame:
    """Create Task 1 repeated-CV tables without modifying legacy outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cohorts = load_cohorts()
    model_data = (
        ("clinical", cohorts.clinical, cohorts.clinical_features),
        ("radiomic", cohorts.final, cohorts.radiomic_features),
        ("combined", cohorts.final, cohorts.combined_features),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    summaries: list[dict[str, float | int | str]] = []
    for name, frame, features in model_data:
        folds, predictions, summary = repeated_cox_cv(
            frame,
            features,
            model_name=name,
            seeds=DEFAULT_SEEDS,
            penalizer=BASELINE_PENALIZER,
        )
        fold_tables.append(folds)
        prediction_tables.append(predictions)
        summaries.append(summary)

    table = pd.DataFrame(summaries)
    table.to_csv(RESULTS_DIR / "c_index_table.csv", index=False)
    pd.concat(fold_tables, ignore_index=True).to_csv(
        RESULTS_DIR / "c_index_fold_scores.csv", index=False
    )
    pd.concat(prediction_tables, ignore_index=True).to_csv(
        RESULTS_DIR / "c_index_oof_predictions.csv", index=False
    )
    headers = list(table.columns)
    rows = [
        [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        for row in table.itertuples(index=False, name=None)
    ]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]
    markdown_text = "\n".join(markdown)
    (RESULTS_DIR / "c_index_table.md").write_text(markdown_text + "\n", encoding="utf-8")
    print(markdown_text)
    return table


def run_task_2() -> pd.DataFrame:
    """Save in-sample and repeated out-of-fold combined-model hazard scores."""
    cohorts = load_cohorts()
    frame = cohorts.final.reset_index(drop=True)
    features = cohorts.combined_features
    fitted = fit_cox_model(
        feature_frame(frame, features),
        frame[TIME_COLUMN],
        frame[EVENT_COLUMN],
        penalizer=BASELINE_PENALIZER,
    )
    in_sample_log_hazard = fitted.predict_log_hazard(
        feature_frame(frame, features)
    ).reset_index(drop=True)

    oof_path = RESULTS_DIR / "c_index_oof_predictions.csv"
    if not oof_path.is_file():
        raise FileNotFoundError(f"Task 1 OOF predictions are required: {oof_path}")
    oof_predictions = pd.read_csv(oof_path)
    combined_oof = oof_predictions.loc[oof_predictions["model"] == "combined"]
    repeated_scores = combined_oof.groupby(ID_COLUMN)["oof_log_hazard"].mean()
    expected_ids = set(frame[ID_COLUMN])
    if set(repeated_scores.index) != expected_ids:
        raise ValueError("Combined OOF predictions do not cover exactly the 52 cohort IDs.")
    oof_log_hazard = repeated_scores.reindex(frame[ID_COLUMN]).reset_index(drop=True)

    in_sample_groups, in_sample_median = median_risk_groups(in_sample_log_hazard)
    oof_groups, oof_median = median_risk_groups(oof_log_hazard)
    scores = frame[[ID_COLUMN, TIME_COLUMN, EVENT_COLUMN]].copy()
    scores["in_sample_log_hazard"] = in_sample_log_hazard
    scores["in_sample_partial_hazard"] = np.exp(in_sample_log_hazard)
    scores["in_sample_risk_group"] = in_sample_groups
    scores["oof_mean_log_hazard"] = oof_log_hazard
    scores["oof_risk_group"] = oof_groups
    scores.to_csv(RESULTS_DIR / "hazard_scores.csv", index=False)

    print(
        "Task 2 risk groups (high includes scores tied at the median):\n"
        f"  In-sample median={in_sample_median:.6f}; "
        f"high n={int((in_sample_groups == 'high').sum())}, "
        f"deaths={int(frame.loc[in_sample_groups == 'high', EVENT_COLUMN].sum())}; "
        f"low n={int((in_sample_groups == 'low').sum())}, "
        f"deaths={int(frame.loc[in_sample_groups == 'low', EVENT_COLUMN].sum())}\n"
        f"  OOF median={oof_median:.6f}; "
        f"high n={int((oof_groups == 'high').sum())}, "
        f"deaths={int(frame.loc[oof_groups == 'high', EVENT_COLUMN].sum())}; "
        f"low n={int((oof_groups == 'low').sum())}, "
        f"deaths={int(frame.loc[oof_groups == 'low', EVENT_COLUMN].sum())}"
    )
    return scores


def run_task_3() -> pd.DataFrame:
    """Create the two-panel Kaplan-Meier figure and its statistics table."""
    scores_path = RESULTS_DIR / "hazard_scores.csv"
    if not scores_path.is_file():
        raise FileNotFoundError(f"Task 2 hazard scores are required: {scores_path}")
    scores = pd.read_csv(scores_path)
    statistics = plot_kaplan_meier_comparison(
        scores,
        RESULTS_DIR / "fig_km.png",
    )
    statistics.to_csv(RESULTS_DIR / "km_statistics.csv", index=False)
    print(statistics.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return statistics


def run_task_4() -> pd.DataFrame:
    """Fit the full combined model and save its exploratory coefficient forest plot."""
    cohorts = load_cohorts()
    frame = cohorts.final
    fitted = fit_cox_model(
        feature_frame(frame, cohorts.combined_features),
        frame[TIME_COLUMN],
        frame[EVENT_COLUMN],
        penalizer=BASELINE_PENALIZER,
    )
    table = plot_top_coefficients(
        fitted.model,
        cohorts.clinical_features,
        RESULTS_DIR / "fig_coefficients.png",
        RESULTS_DIR / "top15_coefficients.csv",
    )
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return table


def run_task_5() -> Path:
    """Save stage and resection-margin curves at the cohort-mean profile."""
    cohorts = load_cohorts()
    frame = cohorts.final
    fitted = fit_cox_model(
        feature_frame(frame, cohorts.combined_features),
        frame[TIME_COLUMN],
        frame[EVENT_COLUMN],
        penalizer=BASELINE_PENALIZER,
    )
    figure_path = RESULTS_DIR / "fig_survival_by_covariate.png"
    plot_mean_covariate_survival(
        fitted,
        frame,
        cohorts.combined_features,
        figure_path,
    )
    print(
        "Task 5 predictions vary stage (0-4) and residual_R0 (0/1); "
        "all other clinical/radiomic predictors are fixed at cohort means."
    )
    return figure_path


def run_task_6() -> pd.DataFrame:
    """Compare nested tuning, fold-local feature selection, PCA, and stage baseline."""
    cohorts = load_cohorts()
    frame = cohorts.final.reset_index(drop=True)
    selection_cache: SelectionCache = {}
    strategies = [
        nested_penalizer_grid_cv(
            frame,
            cohorts.combined_features,
            grid=PENALIZER_GRID,
        ),
        *(
            univariate_cox_top_k_cv(
                frame,
                cohorts.combined_features,
                k,
                selection_cache=selection_cache,
                dataset_scope="final_dataset",
            )
            for k in (3, 5, 10)
        ),
        radiomic_pca_cv(
            frame,
            cohorts.clinical_features,
            cohorts.radiomic_features,
            n_components=5,
        ),
        *(
            clinical_plus_top_k_radiomics_cv(
                frame,
                cohorts.clinical_features,
                cohorts.radiomic_features,
                k,
                selection_cache=selection_cache,
                dataset_scope="final_dataset",
            )
            for k in (3, 5, 10)
        ),
        stage_only_cv(frame),
    ]
    summaries = [summary for summary, _ in strategies]
    folds = pd.concat([fold_table for _, fold_table in strategies], ignore_index=True)
    for model_name, model_folds in folds.groupby("model"):
        assert len(model_folds) == 10 * 5, f"{model_name} must have exactly 50 outer folds."
        assert model_folds.groupby("seed").size().eq(10).all()
        assert model_folds["c_index"].notna().all()
    comparison = pd.DataFrame(summaries)
    comparison.to_csv(RESULTS_DIR / "improvements_comparison.csv", index=False)
    folds.to_csv(RESULTS_DIR / "improvements_fold_scores.csv", index=False)

    headers = list(comparison.columns)
    markdown_rows = [
        [
            (
                f"{value:.3f}"
                if isinstance(value, (float, np.floating)) and np.isfinite(value)
                else ""
                if isinstance(value, (float, np.floating))
                else str(value)
            )
            for value in row
        ]
        for row in comparison.itertuples(index=False, name=None)
    ]
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in markdown_rows),
    ]
    (RESULTS_DIR / "improvements_comparison.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    return comparison


def run_task_7() -> pd.DataFrame:
    """Run and save the optional four-way SMOTE leakage ablation."""
    cohorts = load_cohorts()
    comparison, folds, metadata = run_smote_ablation(
        cohorts.final,
        cohorts.combined_features,
        seeds=DEFAULT_SEEDS,
    )
    comparison.to_csv(RESULTS_DIR / "smote_ablation.csv", index=False)
    folds.to_csv(RESULTS_DIR / "smote_ablation_fold_scores.csv", index=False)
    assumptions = (
        f"Threshold: {metadata['threshold_days']} days; "
        f"excluded early-censored patients: {metadata['n_dropped_censored_before_threshold']}; "
        f"eligible n={metadata['n_eligible_patients']}; "
        f"high-risk={metadata['n_high_risk']}, low-risk={metadata['n_low_risk']}.\n\n"
        "High risk means an observed death before the threshold; low risk means follow-up "
        "at or beyond the threshold. Early-censored patients are excluded because their "
        "threshold status is unknown. SMOTE uses k_neighbors=min(5, minority-class count "
        "minus 1) for each resampling set.\n\n"
        "SMOTE uses standardized clinical/radiomic predictors plus standardized time and "
        "event, then inverse-transforms, rounds event to 0/1, and clips time to at least "
        "one day. In C, SMOTE preprocessing is fitted to each training fold only. In B/A, "
        "whole-cohort SMOTE before CV is the intentional leakage ablation; Cox scaling is "
        "still fitted only to each Cox training fold. B scores real test rows only; A "
        "scores real and synthetic test rows.\n\n"
    )
    markdown_headers = list(comparison.columns)
    markdown_rows = [
        [
            f"{value:.3f}"
            if isinstance(value, (float, np.floating)) and np.isfinite(value)
            else str(value)
            for value in row
        ]
        for row in comparison.itertuples(index=False, name=None)
    ]
    markdown = [
        assumptions,
        "| " + " | ".join(markdown_headers) + " |",
        "| " + " | ".join("---" for _ in markdown_headers) + " |",
        *("| " + " | ".join(row) + " |" for row in markdown_rows),
    ]
    (RESULTS_DIR / "smote_ablation.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    plot_smote_ablation(comparison, RESULTS_DIR / "fig_smote_ablation.png")
    print("Task 7 threshold-label metadata:", metadata)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    inside_score = float(
        comparison.loc[
            comparison["setup"] == "C_train_fold_smote_real_test",
            "mean_repeat_fold_c_index",
        ].iloc[0]
    )
    if inside_score > 0.65:
        raise RuntimeError(
            "Task 7 C exceeded 0.65. Audit training-fold SMOTE and split isolation "
            "before interpreting the ablation."
        )
    return comparison


def main() -> None:
    """Run completed analysis tasks in order."""
    run_task_1()
    run_task_2()
    run_task_3()
    run_task_4()
    run_task_5()
    run_task_6()
    run_task_7()


if __name__ == "__main__":
    main()
