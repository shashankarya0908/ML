"""Load and validate the pancreatic cancer analysis cohorts."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ID_COLUMN = "Case Submitter ID"
TIME_COLUMN = "time"
EVENT_COLUMN = "event"
CLINICAL_FEATURES = (
    "age_years",
    "male",
    "grade",
    "stage",
    "residual_R0",
    "head_of_pancreas",
)


@dataclass(frozen=True)
class Cohorts:
    """Validated clinical and radiomic analysis tables."""

    clinical: pd.DataFrame
    final: pd.DataFrame
    clinical_features: tuple[str, ...]
    radiomic_features: tuple[str, ...]

    @property
    def combined_features(self) -> tuple[str, ...]:
        """Return the allowed clinical and radiomic predictors."""
        return self.clinical_features + self.radiomic_features


def load_cohorts(
    clinical_path: Path | str = PROJECT_ROOT / "clinical_features.csv",
    final_path: Path | str = PROJECT_ROOT / "final_dataset.csv",
) -> Cohorts:
    """Read the supplied cohorts and enforce their expected schema."""
    clinical = pd.read_csv(clinical_path)
    final = pd.read_csv(final_path)
    radiomic_features = tuple(
        column for column in final.columns if column.startswith("original_")
    )

    required = {ID_COLUMN, TIME_COLUMN, EVENT_COLUMN, *CLINICAL_FEATURES}
    for name, frame in (("clinical_features.csv", clinical), ("final_dataset.csv", final)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
        if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} must have unique, non-missing patient IDs.")
        if frame[[TIME_COLUMN, EVENT_COLUMN, *CLINICAL_FEATURES]].isna().any().any():
            raise ValueError(f"{name} has missing outcome or clinical-feature values.")
        if not frame[EVENT_COLUMN].isin((0, 1)).all():
            raise ValueError(f"{name} event values must be 0 (censored) or 1 (death).")
        if (frame[TIME_COLUMN] <= 0).any():
            raise ValueError(f"{name} survival times must be positive.")

    if len(clinical) != 77 or int(clinical[EVENT_COLUMN].sum()) != 65:
        raise ValueError("clinical_features.csv must contain 77 patients and 65 deaths.")
    if len(final) != 52 or int(final[EVENT_COLUMN].sum()) != 44:
        raise ValueError("final_dataset.csv must contain 52 patients and 44 deaths.")
    if len(radiomic_features) != 45:
        raise ValueError(f"Expected 45 radiomic features; found {len(radiomic_features)}.")
    if not set(final[ID_COLUMN]).issubset(set(clinical[ID_COLUMN])):
        raise ValueError("The 52-patient cohort IDs must be a subset of the 77-patient IDs.")
    if final[list(radiomic_features)].isna().any().any():
        raise ValueError("final_dataset.csv has missing radiomic feature values.")

    return Cohorts(
        clinical=clinical,
        final=final,
        clinical_features=CLINICAL_FEATURES,
        radiomic_features=radiomic_features,
    )


def feature_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    """Select only declared predictors, rejecting IDs and outcome leakage."""
    forbidden = {ID_COLUMN, TIME_COLUMN, EVENT_COLUMN}
    forbidden_prefixes = ("days_to_death", "vital_status", "progression", "recurrence", "cause_of_death")
    if forbidden.intersection(features):
        raise ValueError("Patient ID, survival time, and event cannot be predictors.")
    if any(name.lower().startswith(forbidden_prefixes) for name in features):
        raise ValueError("Outcome-derived columns cannot be predictors.")
    if not features:
        raise ValueError("At least one predictor is required.")
    return frame.loc[:, list(features)].copy()
