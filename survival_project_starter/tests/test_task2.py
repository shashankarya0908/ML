"""Tests for combined-model median risk grouping."""

import unittest

import pandas as pd

from src.cv import median_risk_groups


class MedianRiskGroupTests(unittest.TestCase):
    def test_median_ties_are_assigned_to_high_risk(self) -> None:
        scores = pd.Series([0.1, 0.2, 0.2, 0.4])
        groups, median = median_risk_groups(scores)
        self.assertEqual(median, 0.2)
        self.assertEqual(groups.tolist(), ["low", "high", "high", "high"])

    def test_non_finite_or_single_group_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            median_risk_groups(pd.Series([1.0, float("nan")]))
        with self.assertRaises(ValueError):
            median_risk_groups(pd.Series([1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
