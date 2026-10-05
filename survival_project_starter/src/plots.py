"""Plots and descriptive survival summaries."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D

from src.data import EVENT_COLUMN, TIME_COLUMN
from src.models import FittedCox


LOW_RISK_COLOR = "#2b6cb0"
HIGH_RISK_COLOR = "#c53030"


def _split_statistics(
    scores: pd.DataFrame,
    group_column: str,
    label: str,
) -> dict[str, str | float | int]:
    """Calculate log-rank and high-vs-low Cox HR for one median split."""
    high = scores[group_column].eq("high")
    low = scores[group_column].eq("low")
    if not high.any() or not low.any():
        raise ValueError(f"{label} risk split must contain both groups.")

    high_rows = scores.loc[high]
    low_rows = scores.loc[low]
    logrank = logrank_test(
        high_rows[TIME_COLUMN],
        low_rows[TIME_COLUMN],
        event_observed_A=high_rows[EVENT_COLUMN],
        event_observed_B=low_rows[EVENT_COLUMN],
    )
    group_model_data = pd.DataFrame(
        {
            TIME_COLUMN: scores[TIME_COLUMN].to_numpy(),
            EVENT_COLUMN: scores[EVENT_COLUMN].to_numpy(),
            "high_risk": high.astype(int).to_numpy(),
        }
    )
    cox = CoxPHFitter(penalizer=0.0, l1_ratio=0.0)
    cox.fit(group_model_data, duration_col=TIME_COLUMN, event_col=EVENT_COLUMN)
    estimate = cox.summary.loc["high_risk"]
    return {
        "version": label,
        "n_high": int(high.sum()),
        "deaths_high": int(high_rows[EVENT_COLUMN].sum()),
        "n_low": int(low.sum()),
        "deaths_low": int(low_rows[EVENT_COLUMN].sum()),
        "logrank_chi2": float(logrank.test_statistic),
        "logrank_p": float(logrank.p_value),
        "hazard_ratio_high_vs_low": float(estimate["exp(coef)"]),
        "hazard_ratio_ci_lower": float(estimate["exp(coef) lower 95%"]),
        "hazard_ratio_ci_upper": float(estimate["exp(coef) upper 95%"]),
    }


def plot_kaplan_meier_comparison(
    scores: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Save in-sample and OOF KM panels with CIs, censor marks, and group statistics."""
    specifications = (
        ("in_sample_risk_group", "A. In-sample combined-model scores"),
        ("oof_risk_group", "B. Out-of-fold mean scores (five seeds)"),
    )
    statistics = [
        _split_statistics(scores, group_column, label)
        for group_column, label in specifications
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (group_column, title), stats in zip(axes, specifications, statistics):
        for group, color in (("low", LOW_RISK_COLOR), ("high", HIGH_RISK_COLOR)):
            subset = scores.loc[scores[group_column].eq(group)]
            deaths = int(subset[EVENT_COLUMN].sum())
            km = KaplanMeierFitter()
            km.fit(
                subset[TIME_COLUMN],
                event_observed=subset[EVENT_COLUMN],
                label=f"{group.title()} risk (n={len(subset)}, {deaths} deaths)",
            )
            km.plot_survival_function(
                ax=ax,
                ci_show=True,
                ci_alpha=0.16,
                color=color,
                show_censors=True,
                censor_styles={"marker": "|", "ms": 6},
            )
        p_text = f"{stats['logrank_p']:.2e}" if stats["logrank_p"] < 0.001 else f"{stats['logrank_p']:.3f}"
        ax.text(
            0.04,
            0.06,
            f"Log-rank $\\chi^2$={stats['logrank_chi2']:.2f}, p={p_text}\n"
            f"HR (high vs low)={stats['hazard_ratio_high_vs_low']:.2f} "
            f"[{stats['hazard_ratio_ci_lower']:.2f}, {stats['hazard_ratio_ci_upper']:.2f}]",
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Survival time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    fig.suptitle("Combined Cox model: median-split Kaplan-Meier curves", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(statistics)


def plot_top_coefficients(
    model: CoxPHFitter,
    clinical_features: tuple[str, ...],
    figure_path: Path,
    table_path: Path,
) -> pd.DataFrame:
    """Plot and save the 15 largest absolute combined-model log-HR estimates."""
    summary = model.summary.copy()
    top = summary.assign(abs_coef=summary["coef"].abs()).nlargest(15, "abs_coef")
    top = top.sort_values("coef").copy()
    top["feature"] = top.index
    top["feature_group"] = [
        "clinical" if feature in clinical_features else "radiomic"
        for feature in top.index
    ]
    top["ci_lower"] = top["coef lower 95%"]
    top["ci_upper"] = top["coef upper 95%"]
    top["ci_includes_zero"] = (top["ci_lower"] <= 0) & (top["ci_upper"] >= 0)
    result = top[
        ["feature", "feature_group", "coef", "ci_lower", "ci_upper", "ci_includes_zero"]
    ].reset_index(drop=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(table_path, index=False)

    clinical_color = "#dd6b20"
    radiomic_color = "#2b6cb0"
    fig, ax = plt.subplots(figsize=(10, 7))
    for position, row in result.iterrows():
        color = clinical_color if row["feature_group"] == "clinical" else radiomic_color
        ax.errorbar(
            row["coef"],
            position,
            xerr=np.array(
                [[row["coef"] - row["ci_lower"]], [row["ci_upper"] - row["coef"]]]
            ),
            fmt="s",
            color=color,
            markerfacecolor="white" if row["ci_includes_zero"] else color,
            markeredgecolor=color,
            capsize=3,
            markersize=7,
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(len(result)))
    ax.set_yticklabels(
        [str(feature).removeprefix("original_") for feature in result["feature"]],
        fontsize=8.5,
    )
    ax.set_xlabel("Log hazard ratio per 1 SD increase (95% CI)")
    ax.set_title("Top 15 combined-model Cox coefficients", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    handles = (
        Line2D([], [], marker="s", color=clinical_color, linestyle="None", label="Clinical"),
        Line2D([], [], marker="s", color=radiomic_color, linestyle="None", label="Radiomic"),
        Line2D(
            [],
            [],
            marker="s",
            color="gray",
            markerfacecolor="white",
            linestyle="None",
            label="95% CI includes 0",
        ),
    )
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def mean_covariate_profiles(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    varying_feature: str,
) -> pd.DataFrame:
    """Build profiles at cohort means, changing only the requested feature."""
    if varying_feature not in features:
        raise ValueError(f"{varying_feature} is not a model feature.")
    values = sorted(frame[varying_feature].unique())
    means = frame.loc[:, list(features)].mean()
    profiles = pd.DataFrame(
        np.repeat(means.to_numpy()[None, :], len(values), axis=0),
        columns=features,
    )
    profiles[varying_feature] = values
    return profiles


def plot_mean_covariate_survival(
    fitted: FittedCox,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    figure_path: Path,
) -> None:
    """Plot predicted survival while varying stage and residual_R0 at cohort means."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    specifications = (
        ("stage", axes[0]),
        ("residual_R0", axes[1]),
    )
    stage_names = {0: "IB", 1: "IIA", 2: "IIB", 3: "III", 4: "IV"}

    for feature, ax in specifications:
        profiles = mean_covariate_profiles(frame, features, feature)
        survival = fitted.model.predict_survival_function(fitted.transform(profiles))
        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.88, len(profiles)))
        for position, profile in profiles.iterrows():
            value = profile[feature]
            count = int(frame[feature].eq(value).sum())
            if feature == "stage":
                label = f"Stage {stage_names.get(int(value), str(value))} (n={count})"
            else:
                margin = "clear margins" if int(value) == 1 else "positive margins"
                label = f"residual_R0={int(value)} ({margin}, n={count})"
            ax.step(
                survival.index,
                survival.iloc[:, position],
                where="post",
                color=colors[position],
                label=label,
                linewidth=2,
            )
        ax.set_title(f"Vary {feature}; all others at cohort mean", loc="left", fontweight="bold")
        ax.set_xlabel("Survival time (days)")
        ax.set_ylabel("Predicted survival probability")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, frameon=False)

    fig.suptitle("Combined Cox model: adjusted survival profiles", fontsize=13)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_smote_ablation(
    comparison: pd.DataFrame,
    figure_path: Path,
) -> None:
    """Plot repeated-CV mean C-index for the four SMOTE leakage setups."""
    labels = {
        "D_no_smote_real_test": "D\nNo SMOTE",
        "C_train_fold_smote_real_test": "C\nTrain-fold SMOTE",
        "B_pre_cv_smote_real_test": "B\nPre-CV SMOTE\n(real test)",
        "A_pre_cv_smote_all_test_rows": "A\nPre-CV SMOTE\n(all test rows)",
    }
    data = comparison.copy()
    colors = ["#718096", "#2b6cb0", "#dd6b20", "#c53030"]
    positions = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(
        positions,
        data["mean_repeat_fold_c_index"],
        yerr=data["sd_repeat_fold_c_index"],
        color=colors[:len(data)],
        capsize=5,
        edgecolor="white",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[name] for name in data["setup"]])
    ax.set_ylabel("Mean per-fold C-index (mean ± SD over 5 seeds)")
    ax.set_ylim(0, 1)
    ax.set_title("SMOTE leakage ablation: threshold-based risk labels", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
