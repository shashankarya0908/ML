"""Tests for mean-profile predicted survival curves."""

import unittest

import numpy as np

from src.plots import mean_covariate_profiles
from src.data import CLINICAL_FEATURES, load_cohorts


class MeanProfileTests(unittest.TestCase):
    def test_only_the_requested_feature_varies(self) -> None:
        cohorts = load_cohorts()
        profiles = mean_covariate_profiles(
            cohorts.final,
            CLINICAL_FEATURES + cohorts.radiomic_features,
            "stage",
        )
        self.assertEqual(profiles["stage"].tolist(), [0, 1, 2, 3, 4])
        other_features = [
            feature
            for feature in CLINICAL_FEATURES + cohorts.radiomic_features
            if feature != "stage"
        ]
        np.testing.assert_allclose(
            profiles[other_features].to_numpy(),
            np.tile(
                cohorts.final[other_features].mean().to_numpy(),
                (len(profiles), 1),
            ),
        )


if __name__ == "__main__":
    unittest.main()
