"""Focused tests for the input guards and Task 1 cross-validation."""

import unittest

import numpy as np

from lifelines.utils import concordance_index

from src.cv import c_index_from_log_hazard
from src.data import (
    CLINICAL_FEATURES,
    EVENT_COLUMN,
    ID_COLUMN,
    TIME_COLUMN,
    feature_frame,
    load_cohorts,
)
from src.models import fit_cox_model


class CohortValidationTests(unittest.TestCase):
    def test_expected_cohort_counts_and_feature_counts(self) -> None:
        cohorts = load_cohorts()
        self.assertEqual(len(cohorts.final), 52)
        self.assertEqual(len(cohorts.clinical), 77)
        self.assertEqual(int(cohorts.final[EVENT_COLUMN].sum()), 44)
        self.assertEqual(int(cohorts.clinical[EVENT_COLUMN].sum()), 65)
        self.assertEqual(len(CLINICAL_FEATURES), 6)
        self.assertEqual(len(cohorts.radiomic_features), 45)

    def test_small_cohort_ids_are_subset_and_features_exclude_identifiers_and_outcomes(self) -> None:
        cohorts = load_cohorts()
        self.assertTrue(
            set(cohorts.final[ID_COLUMN]).issubset(set(cohorts.clinical[ID_COLUMN]))
        )
        for frame, features in (
            (cohorts.clinical, cohorts.clinical_features),
            (cohorts.final, cohorts.radiomic_features),
            (cohorts.final, cohorts.combined_features),
        ):
            selected = feature_frame(frame, features)
            self.assertNotIn(ID_COLUMN, selected.columns)
            self.assertNotIn(TIME_COLUMN, selected.columns)
            self.assertNotIn(EVENT_COLUMN, selected.columns)
            self.assertTrue(all(column.startswith("original_") for column in features
                                if column not in CLINICAL_FEATURES))


class ConcordanceAndScalingTests(unittest.TestCase):
    def test_perfectly_ordered_synthetic_survival_has_c_index_one(self) -> None:
        event_times = np.array([1.0, 2.0, 3.0, 4.0])
        events = np.ones(4, dtype=int)
        longer_survival_scores = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            concordance_index(event_times, longer_survival_scores, events),
            1.0,
        )
        higher_risk_log_hazard = -longer_survival_scores
        self.assertEqual(
            c_index_from_log_hazard(event_times, higher_risk_log_hazard, events),
            1.0,
        )

    def test_scaler_is_fit_on_training_rows_only(self) -> None:
        cohorts = load_cohorts()
        features = ("age_years", "stage")
        train = cohorts.final.iloc[:40]
        fitted = fit_cox_model(
            feature_frame(train, features),
            train[TIME_COLUMN],
            train[EVENT_COLUMN],
        )
        np.testing.assert_allclose(
            fitted.scaler.mean_,
            train[list(features)].mean().to_numpy(),
        )
        self.assertFalse(
            np.allclose(
                fitted.scaler.mean_,
                cohorts.final[list(features)].mean().to_numpy(),
            )
        )


if __name__ == "__main__":
    unittest.main()
