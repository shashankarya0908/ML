"""Cox proportional-hazards model fitting utilities."""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from src.data import EVENT_COLUMN, ID_COLUMN, TIME_COLUMN


LOGGER = logging.getLogger(__name__)
BASELINE_PENALIZER = 0.1


@dataclass
class FittedCox:
    """A Cox model paired with its training-fold scaler."""

    model: CoxPHFitter
    scaler: StandardScaler
    features: tuple[str, ...]
    penalizer: float
    fit_attempts: int
    non_converged_attempts: int

    @property
    def penalizer_escalated(self) -> bool:
        """Whether at least one failed convergence attempt required escalation."""
        return self.fit_attempts > 1

    def transform(self, predictors: pd.DataFrame) -> pd.DataFrame:
        """Scale predictors using statistics learned from the training fold."""
        if tuple(predictors.columns) != self.features:
            raise ValueError("Predictor columns and order must match the fitted model.")
        scaled = self.scaler.transform(predictors)
        return pd.DataFrame(scaled, columns=self.features, index=predictors.index)

    def predict_log_hazard(self, predictors: pd.DataFrame) -> pd.Series:
        """Return the Cox log partial hazard (higher means higher risk)."""
        return self.model.predict_log_partial_hazard(self.transform(predictors))


def fit_cox_model(
    predictors: pd.DataFrame,
    times: pd.Series,
    events: pd.Series,
    penalizer: float = BASELINE_PENALIZER,
) -> FittedCox:
    """Fit a Cox model after training-only scaling, escalating penalty on convergence."""
    forbidden = {ID_COLUMN, TIME_COLUMN, EVENT_COLUMN}.intersection(predictors.columns)
    if forbidden:
        raise ValueError(f"Forbidden model predictors: {sorted(forbidden)}")
    if predictors.empty or len(predictors) != len(times) or len(times) != len(events):
        raise ValueError("Predictors, times, and events must have matching non-empty rows.")
    if penalizer <= 0:
        raise ValueError("The Cox penalizer must be positive.")

    features = tuple(predictors.columns)
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(predictors),
        columns=features,
        index=predictors.index,
    )
    training = scaled.copy()
    training[TIME_COLUMN] = np.asarray(times, dtype=float)
    training[EVENT_COLUMN] = np.asarray(events, dtype=int)

    last_issue: str | None = None
    fit_attempts = 0
    non_converged_attempts = 0
    for multiplier in (1.0, 10.0, 100.0):
        current_penalizer = penalizer * multiplier
        model = CoxPHFitter(penalizer=current_penalizer, l1_ratio=0.0)
        fit_attempts += 1
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(
                    training,
                    duration_col=TIME_COLUMN,
                    event_col=EVENT_COLUMN,
                    show_progress=False,
                )
            convergence_warnings = [
                item for item in caught
                if issubclass(item.category, ConvergenceWarning)
            ]
            for item in caught:
                if not issubclass(item.category, ConvergenceWarning):
                    warnings.warn(item.message, item.category)
            if convergence_warnings:
                non_converged_attempts += 1
                last_issue = "; ".join(str(item.message) for item in convergence_warnings)
                LOGGER.debug(
                    "Cox fit emitted a convergence warning at penalizer %g; "
                    "retrying with a higher penalty: %s",
                    current_penalizer,
                    last_issue,
                )
                continue
            if multiplier > 1.0:
                LOGGER.debug(
                    "Cox fit converged after raising penalizer from %g to %g.",
                    penalizer,
                    current_penalizer,
                )
            return FittedCox(
                model,
                scaler,
                features,
                current_penalizer,
                fit_attempts,
                non_converged_attempts,
            )
        except ConvergenceError as exc:
            non_converged_attempts += 1
            last_issue = str(exc)
            LOGGER.debug(
                "Cox fit failed at penalizer %g; retrying with a higher penalty: %s",
                current_penalizer,
                exc,
            )

    raise RuntimeError(
        f"Cox model failed to converge after penalizer escalation from {penalizer:g}: "
        f"{last_issue or 'unknown convergence failure'}"
    )
