"""Focused validation for Task 3 survival plots."""

import unittest

import pandas as pd

from src.data import EVENT_COLUMN, TIME_COLUMN
from src.plots import _split_statistics


class KaplanMeierStatisticsTests(unittest.TestCase):
    def test_group_counts_and_hazard_ratio_are_reported(self) -> None:
        scores = pd.DataFrame(
            {
                TIME_COLUMN: [10, 12, 14, 16, 22, 24, 26, 28],
                EVENT_COLUMN: [1, 1, 0, 0, 1, 0, 0, 1],
                "risk": ["high", "low", "high", "low", "high", "low", "high", "low"],
            }
        )
        result = _split_statistics(scores, "risk", "synthetic")
        self.assertEqual(result["n_high"], 4)
        self.assertEqual(result["deaths_high"], 2)
        self.assertEqual(result["n_low"], 4)
        self.assertEqual(result["deaths_low"], 2)
        self.assertGreater(result["hazard_ratio_high_vs_low"], 0)
        self.assertGreaterEqual(result["hazard_ratio_ci_upper"], result["hazard_ratio_ci_lower"])


if __name__ == "__main__":
    unittest.main()
