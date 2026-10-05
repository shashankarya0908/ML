"""
Pancreatic Cancer Survival Analysis
Models: Clinical, Radiomic, Combined Cox Proportional Hazards
Outputs: CV C-index, Kaplan-Meier curves, log-rank test,
top 15 coefficients, and adjusted survival curves.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from sklearn.model_selection import KFold

from src.models import FittedCox, fit_cox_model


# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

CLINICAL_FILE = BASE_DIR / "clinical_features.csv"
FINAL_FILE = BASE_DIR / "final_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

ID_COL = "Case Submitter ID"
TIME_COL = "time"
EVENT_COL = "event"

CLINICAL_FEATURES = [
    "age_years",
    "male",
    "grade",
    "stage",
    "residual_R0",
    "head_of_pancreas"
]

# Stronger shrinkage stabilizes the high-dimensional 52-patient Cox fits.
PENALIZER = 1.0
N_FOLDS = 10
RANDOM_SEED = 42


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

def load_and_validate():

    if not CLINICAL_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {CLINICAL_FILE.name}. "
            "Place it in the same folder as this script."
        )

    if not FINAL_FILE.exists():
        raise FileNotFoundError(
            f"Cannot find {FINAL_FILE.name}. "
            "Place it in the same folder as this script."
        )

    clinical = pd.read_csv(CLINICAL_FILE)
    final = pd.read_csv(FINAL_FILE)

    # Remove accidental whitespace in column names.
    clinical.columns = clinical.columns.str.strip()
    final.columns = final.columns.str.strip()

    radiomic_features = [
        col for col in final.columns
        if col.startswith("original_")
    ]

    required = {
        ID_COL,
        TIME_COL,
        EVENT_COL,
        *CLINICAL_FEATURES
    }

    for name, data in [
        ("clinical_features.csv", clinical),
        ("final_dataset.csv", final)
    ]:
        missing = required - set(data.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    if not radiomic_features:
        raise ValueError(
            "No columns starting with 'original_' "
            "were found in final_dataset.csv."
        )

    if clinical[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs in clinical_features.csv.")

    if final[ID_COL].duplicated().any():
        raise ValueError("Duplicate IDs in final_dataset.csv.")

    if not set(final[ID_COL]).issubset(set(clinical[ID_COL])):
        raise ValueError(
            "Patient IDs in final_dataset.csv must be present "
            "in clinical_features.csv."
        )

    for name, data in [
        ("clinical", clinical),
        ("final", final)
    ]:
        data[TIME_COL] = pd.to_numeric(
            data[TIME_COL], errors="raise"
        )
        data[EVENT_COL] = pd.to_numeric(
            data[EVENT_COL], errors="raise"
        )

        if data[[TIME_COL, EVENT_COL]].isna().any().any():
            raise ValueError(
                f"Missing survival targets in {name} data."
            )

        if not data[EVENT_COL].isin([0, 1]).all():
            raise ValueError(
                f"{name}: event must be 1 for death "
                "and 0 for censored observations."
            )

        if (data[TIME_COL] <= 0).any():
            raise ValueError(
                f"{name}: survival times must be positive."
            )

    if clinical[CLINICAL_FEATURES].isna().any().any():
        raise ValueError(
            "Clinical predictors contain missing values."
        )

    if final[CLINICAL_FEATURES + radiomic_features].isna().any().any():
        raise ValueError(
            "Predictors in final_dataset.csv contain missing values. "
            "Check the input data before proceeding."
        )

    print("\nDATASET SUMMARY")
    print("-" * 45)

    print(
        f"Clinical: {len(clinical)} patients, "
        f"{int(clinical[EVENT_COL].sum())} deaths"
    )

    print(
        f"Radiomic/combined: {len(final)} patients, "
        f"{len(radiomic_features)} radiomic features, "
        f"{int(final[EVENT_COL].sum())} deaths"
    )

    return clinical, final, radiomic_features


# --------------------------------------------------
# 3. COX MODEL
# --------------------------------------------------

def fit_cox(data, features, penalizer=PENALIZER) -> FittedCox:
    """Fit on standardized predictors and retry with stronger regularization if needed."""
    return fit_cox_model(
        data.loc[:, list(features)],
        data[TIME_COL],
        data[EVENT_COL],
        penalizer=penalizer,
    )


# --------------------------------------------------
# 4. TEN-FOLD CROSS-VALIDATION
# --------------------------------------------------

def cv_cindex(
    data,
    features,
    model_name,
    n_splits=N_FOLDS,
    penalizer=PENALIZER
):

    X = data[features].astype(float).to_numpy()
    times = data[TIME_COL].astype(float).to_numpy()
    events = data[EVENT_COL].astype(int).to_numpy()

    splitter = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X), start=1
    ):

        print(f"{model_name}: fold {fold}/{n_splits}")

        X_train = X[train_idx]
        X_test = X[test_idx]

        t_train = times[train_idx]
        e_train = events[train_idx]

        train_data = pd.DataFrame(
            X_train,
            columns=features
        )
        test_data = pd.DataFrame(
            X_test,
            columns=features
        )

        score = np.nan
        note = ""
        
        try:
            fitted = fit_cox_model(
                train_data,
                pd.Series(t_train),
                pd.Series(e_train),
                penalizer=penalizer,
            )
            risk = fitted.predict_log_hazard(test_data).to_numpy().ravel()

            # Higher Cox risk means shorter predicted survival.
            score = concordance_index(
                times[test_idx],
                -risk,
                events[test_idx]
            )

        except Exception as exc:
            note = str(exc)
            print(f"Warning: fold {fold} failed: {note}")

        fold_results.append({
            "model": model_name,
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "test_events": int(events[test_idx].sum()),
            "c_index": score,
            "note": note
        })

    result = pd.DataFrame(fold_results)

    result.to_csv(
        OUTPUT_DIR / f"{model_name}_cv_folds.csv",
        index=False
    )

    return result


def summarize_cv(model_name, result):

    scores = result["c_index"].dropna()

    return {
        "model": model_name,
        "valid_folds": len(scores),
        "mean_c_index": scores.mean(),
        "sd_fold_c_index": scores.std(ddof=1),
        "min_fold_c_index": scores.min(),
        "max_fold_c_index": scores.max()
    }


# --------------------------------------------------
# 5. KAPLAN-MEIER AND LOG-RANK TEST
# --------------------------------------------------

def plot_km_from_combined(fitted, data, features):

    # IMPORTANT:
    # Keep ID_COL in data for saving patient results.
    # Pass only model features; fitted.transform applies training-fold scaling.

    scored = data.copy()

    scored["hazard_score"] = (
        np.exp(fitted.predict_log_hazard(scored[features]).to_numpy().ravel())
    )

    median_score = scored["hazard_score"].median()

    scored["risk_group"] = np.where(
        scored["hazard_score"] >= median_score,
        "High risk",
        "Low risk"
    )

    # Patient ID is included in the output, never in prediction.
    scored[
        [
            ID_COL,
            TIME_COL,
            EVENT_COL,
            "hazard_score",
            "risk_group"
        ]
    ].to_csv(
        OUTPUT_DIR / "combined_hazard_scores.csv",
        index=False
    )

    km = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(9, 6))

    for group in ["Low risk", "High risk"]:

        subset = scored[
            scored["risk_group"] == group
        ]

        km.fit(
            durations=subset[TIME_COL],
            event_observed=subset[EVENT_COL],
            label=f"{group} (n={len(subset)})"
        )

        km.plot_survival_function(
            ax=ax,
            ci_show=True
        )

    ax.set_title(
        "Kaplan-Meier Curves: Combined Cox Model"
    )

    ax.set_xlabel("Survival time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "kaplan_meier_high_low_risk.png",
        dpi=180
    )

    plt.close(fig)

    # Log-rank test.
    high = scored[scored["risk_group"] == "High risk"]
    low = scored[scored["risk_group"] == "Low risk"]

    test = logrank_test(
        high[TIME_COL],
        low[TIME_COL],
        event_observed_A=high[EVENT_COL],
        event_observed_B=low[EVENT_COL]
    )

    report = (
        f"Median hazard score: {median_score:.6f}\n"
        f"High-risk patients: {len(high)}\n"
        f"High-risk deaths: {int(high[EVENT_COL].sum())}\n"
        f"Low-risk patients: {len(low)}\n"
        f"Low-risk deaths: {int(low[EVENT_COL].sum())}\n"
        f"Log-rank statistic: {test.test_statistic:.6f}\n"
        f"P-value: {test.p_value:.6g}\n"
        "Note: these risk groups use in-sample predictions.\n"
    )

    (
        OUTPUT_DIR / "logrank_test.txt"
    ).write_text(report, encoding="utf-8")

    print("\nLOG-RANK TEST")
    print(report)

    return scored


# --------------------------------------------------
# 6. TOP 15 COEFFICIENT PLOT
# --------------------------------------------------

def plot_top_coefficients(fitted):

    summary = fitted.model.summary.copy()

    summary = summary[
        [
            "coef",
            "coef lower 95%",
            "coef upper 95%"
        ]
    ]

    summary["abs_coef"] = summary["coef"].abs()

    top = summary.nlargest(
        min(15, len(summary)),
        "abs_coef"
    ).sort_values("coef")

    fig, ax = plt.subplots(
        figsize=(11, max(6, 0.4 * len(top)))
    )

    y = np.arange(len(top))

    errors = np.vstack([
        top["coef"] - top["coef lower 95%"],
        top["coef upper 95%"] - top["coef"]
    ])

    ax.errorbar(
        top["coef"],
        y,
        xerr=errors,
        fmt="o",
        capsize=3
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top.index)

    ax.set_xlabel("Cox coefficient per 1 SD (log hazard ratio)")
    ax.set_title("Top 15 Combined-Model Coefficients")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "combined_top15_coefficients.png",
        dpi=180
    )

    plt.close(fig)

    top.drop(columns="abs_coef").to_csv(
        OUTPUT_DIR / "combined_top15_coefficients.csv"
    )


# --------------------------------------------------
# 7. SURVIVAL CURVES FOR VARYING COVARIATES
# --------------------------------------------------

def plot_covariate_survival(fitted, data, features):

    # Keep all predictors in the right order.
    # Other predictors are held at their cohort medians.

    baseline = data[features].median(
        numeric_only=True
    ).to_frame().T

    curve_specs = []

    if "stage" in features:

        stages = sorted(
            pd.to_numeric(data["stage"])
            .dropna()
            .unique()
        )

        curve_specs.append(("stage", stages))

    if "residual_R0" in features:
        curve_specs.append(("residual_R0", [0, 1]))

    if "age_years" in features:

        ages = pd.to_numeric(
            data["age_years"]
        ).dropna()

        curve_specs.append((
            "age_years",
            [
                float(ages.quantile(0.25)),
                float(ages.quantile(0.50)),
                float(ages.quantile(0.75))
            ]
        ))

    for covariate, values in curve_specs:

        fig, ax = plt.subplots(figsize=(9, 6))

        for value in values:

            profile = baseline.copy()
            profile.loc[profile.index[0], covariate] = value

            survival_function = fitted.model.predict_survival_function(
                fitted.transform(profile[features])
            )

            ax.plot(
                survival_function.index,
                survival_function.iloc[:, 0],
                label=f"{covariate} = {value:g}"
            )

        ax.set_title(
            f"Adjusted Predicted Survival by {covariate}"
        )

        ax.set_xlabel("Survival time (days)")
        ax.set_ylabel("Predicted survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        fig.savefig(
            OUTPUT_DIR / f"survival_by_{covariate}.png",
            dpi=180
        )

        plt.close(fig)


# --------------------------------------------------
# 8. MAIN PIPELINE
# --------------------------------------------------

def main():

    clinical, final, radiomic_features = load_and_validate()

    # Clinical model: all 77 patients.
    clinical_data = clinical[
        [
            ID_COL,
            *CLINICAL_FEATURES,
            TIME_COL,
            EVENT_COL
        ]
    ].copy()

    # Radiomic model: only the 52 patients.
    radiomic_data = final[
        [
            ID_COL,
            *radiomic_features,
            TIME_COL,
            EVENT_COL
        ]
    ].copy()

    # Combined model: the same 52 patients.
    combined_features = CLINICAL_FEATURES + radiomic_features

    combined_data = final[
        [
            ID_COL,
            *combined_features,
            TIME_COL,
            EVENT_COL
        ]
    ].copy()

    # Fit descriptive models on the full permitted datasets.
    print("\nFitting Cox models...")

    clinical_model = fit_cox(
        clinical_data,
        CLINICAL_FEATURES
    )

    radiomic_model = fit_cox(
        radiomic_data,
        radiomic_features
    )

    combined_model = fit_cox(
        combined_data,
        combined_features
    )

    clinical_model.model.summary.to_csv(
        OUTPUT_DIR / "clinical_cox_summary.csv"
    )

    radiomic_model.model.summary.to_csv(
        OUTPUT_DIR / "radiomic_cox_summary.csv"
    )

    combined_model.model.summary.to_csv(
        OUTPUT_DIR / "combined_cox_summary.csv"
    )

    # Cross-validation.
    print("\nRunning 10-fold cross-validation...")

    cv_results = []

    model_configs = [
        ("clinical", clinical_data, CLINICAL_FEATURES),
        ("radiomic", radiomic_data, radiomic_features),
        ("combined", combined_data, combined_features)
    ]

    for name, data, features in model_configs:

        folds = cv_cindex(
            data,
            features,
            model_name=name,
            n_splits=N_FOLDS
        )

        cv_results.append(
            summarize_cv(name, folds)
        )

    cv_summary = pd.DataFrame(cv_results)

    cv_summary.to_csv(
        OUTPUT_DIR / "cv_c_index_summary.csv",
        index=False
    )

    print("\nCROSS-VALIDATION RESULTS")
    print(cv_summary.to_string(index=False))

    # Plots and patient hazard scores.
    print("\nCreating plots and risk groups...")

    # CRITICAL FIX:
    # Do NOT drop ID_COL here. Keep it for the output CSV.
    # The prediction functions receive only the feature columns.

    plot_data = combined_data.copy()

    plot_km_from_combined(
        combined_model,
        plot_data,
        combined_features
    )

    plot_top_coefficients(combined_model)

    plot_covariate_survival(
        combined_model,
        plot_data,
        combined_features
    )

    limitations = """
INTERPRETATION AND LIMITATIONS

1. The radiomic and combined models use only 52 patients.
2. The cohort has many predictors relative to its sample size.
3. Small cohorts with many predictors can still produce unstable coefficients,
   even when the regularized model converges.
4. Ten-fold cross-validation scores can vary substantially because each
   test fold contains only about five or six patients.
5. Some folds may have too few comparable patient pairs for a reliable
   C-index.
6. The full-data Cox coefficient plot and Kaplan-Meier risk split are
   exploratory and are not independent validation.
7. The median hazard split is not a clinical decision threshold.
8. Patient IDs are identifiers, not model predictors.
9. Do not include days to death, vital status, progression/recurrence,
   or cause of death as predictors.
10. These results are for research and education, not medical decisions.
"""

    (
        OUTPUT_DIR / "limitations.txt"
    ).write_text(
        limitations,
        encoding="utf-8"
    )

    print("\nSUCCESS: Analysis finished.")
    print(f"All results saved in:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()