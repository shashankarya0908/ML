"""Tests for Task 6 fold-local feature ranking and fold accounting."""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_cohorts
from src.cv import stratified_fold_indices
from src.improvements import (
    SelectionCache,
    _cached_univariate_order,
    select_univariate_cox_features,
    stage_only_cv,
)


class UnivariateRankingTests(unittest.TestCase):
    def test_ranking_fits_separate_univariate_cox_models_on_training_features(self) -> None:
        cohorts = load_cohorts()
        train = cohorts.final.iloc[:40]
        features = ("age_years", "stage", "grade")

        def fake_fit(predictors, times, events, penalizer):
            self.assertEqual(len(predictors.columns), 1)
            self.assertEqual(len(times), len(train))
            self.assertEqual(len(events), len(train))
            self.assertEqual(penalizer, 0.1)
            feature = predictors.columns[0]
            summary = pd.DataFrame(
                {"p": [0.1], "coef": [0.5]},
                index=[feature],
            )
            return SimpleNamespace(model=SimpleNamespace(summary=summary))

        with patch("src.improvements.fit_cox_model", side_effect=fake_fit) as mocked_fit:
            selected = select_univariate_cox_features(train, features, k=2)
        self.assertEqual(mocked_fit.call_count, len(features))
        self.assertEqual(selected, ("age_years", "grade"))

    def test_cache_key_separates_dataset_seed_and_fold(self) -> None:
        cohorts = load_cohorts()
        train = cohorts.final.iloc[:40]
        features = ("age_years", "stage")
        cache: SelectionCache = {}
        with patch(
            "src.improvements.select_univariate_cox_features",
            return_value=features,
        ) as selector:
            for scope, seed, fold in (
                ("final_dataset", 17, 1),
                ("final_dataset", 17, 1),
                ("clinical_features", 17, 1),
                ("final_dataset", 42, 1),
                ("final_dataset", 17, 2),
            ):
                _cached_univariate_order(
                    train,
                    features,
                    seed,
                    fold,
                    scope,
                    cache,
                )
        self.assertEqual(selector.call_count, 4)
        self.assertEqual(len(cache), 4)
        self.assertTrue(all(len(key) == 4 for key in cache))


class FoldAccountingTests(unittest.TestCase):
    def test_small_minority_class_is_distributed_without_sklearn_warning(self) -> None:
        labels = [1] * 44 + [0] * 8
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            folds = list(stratified_fold_indices(labels, n_splits=10, random_state=17))

        with warnings.catch_warnings(record=True) as reference_warnings:
            warnings.simplefilter("always")
            expected_folds = list(
                StratifiedKFold(n_splits=10, shuffle=True, random_state=17).split(
                    np.zeros(len(labels)), labels
                )
            )

        self.assertEqual(caught, [])
        self.assertEqual(len(reference_warnings), 1)
        self.assertIn("least populated class", str(reference_warnings[0].message))
        self.assertEqual(len(folds), 10)
        for actual, expected in zip(folds, expected_folds):
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(
            sorted(len(test_indices) for _, test_indices in folds),
            [5, 5, 5, 5, 5, 5, 5, 5, 6, 6],
        )
        self.assertEqual(
            sorted(int(np.sum(np.asarray(labels)[test_indices] == 0))
                   for _, test_indices in folds),
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        )

    def test_stage_only_has_exactly_fifty_complete_outer_folds(self) -> None:
        cohorts = load_cohorts()
        summary, folds = stage_only_cv(cohorts.final)
        self.assertEqual(summary["n_seeds"], 5)
        self.assertEqual(summary["n_folds_per_seed"], 10)
        self.assertEqual(len(folds), 50)
        self.assertTrue(folds.groupby("seed").size().eq(10).all())
        self.assertTrue(folds["c_index"].notna().all())


if __name__ == "__main__":
    unittest.main()
