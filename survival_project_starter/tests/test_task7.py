"""Tests for the threshold labels and survival-aware SMOTE input construction."""

import unittest

import numpy as np

from src.data import EVENT_COLUMN, ID_COLUMN, TIME_COLUMN, load_cohorts
from src.smote_ablation import (
    IS_REAL_COLUMN,
    RISK_LABEL_COLUMN,
    prepare_smote_ablation_cohort,
    smote_resample_survival_rows,
)


class SmoteAblationTests(unittest.TestCase):
    def test_threshold_labels_drop_only_early_censored_patients(self) -> None:
        cohorts = load_cohorts()
        frame = cohorts.final
        eligible, labels, dropped = prepare_smote_ablation_cohort(
            frame,
            cohorts.combined_features,
        )
        expected_drop = int(
            (frame[EVENT_COLUMN].eq(0) & frame[TIME_COLUMN].lt(270)).sum()
        )
        self.assertEqual(dropped, expected_drop)
        self.assertEqual(len(eligible) + dropped, len(frame))
        self.assertTrue(
            (labels == (
                eligible[EVENT_COLUMN].eq(1) & eligible[TIME_COLUMN].lt(270)
            ).astype(int)).all()
        )
        self.assertEqual(labels.nunique(), 2)

    def test_smote_preserves_original_rows_and_valid_survival_columns(self) -> None:
        cohorts = load_cohorts()
        eligible, labels, _ = prepare_smote_ablation_cohort(
            cohorts.final,
            cohorts.combined_features,
        )
        augmented = smote_resample_survival_rows(
            eligible,
            labels,
            cohorts.combined_features,
            random_state=17,
        )
        self.assertGreater(len(augmented), len(eligible))
        self.assertTrue(augmented[IS_REAL_COLUMN].iloc[:len(eligible)].all())
        self.assertFalse(augmented[IS_REAL_COLUMN].iloc[len(eligible):].any())
        self.assertEqual(
            augmented[ID_COLUMN].iloc[:len(eligible)].tolist(),
            eligible[ID_COLUMN].tolist(),
        )
        np.testing.assert_allclose(
            augmented[[TIME_COLUMN, EVENT_COLUMN]].iloc[:len(eligible)].to_numpy(),
            eligible[[TIME_COLUMN, EVENT_COLUMN]].to_numpy(),
        )
        self.assertTrue(augmented[TIME_COLUMN].ge(1).all())
        self.assertTrue(augmented[EVENT_COLUMN].isin((0, 1)).all())
        self.assertTrue(augmented[RISK_LABEL_COLUMN].isin((0, 1)).all())


if __name__ == "__main__":
    unittest.main()
