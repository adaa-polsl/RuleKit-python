import os
import threading
import unittest

import numpy as np
import pandas as pd

from rulekit import survival
from rulekit.arff import read_arff
from rulekit.events import RuleInductionProgressListener
from rulekit.kaplan_meier import KaplanMeierEstimator
from rulekit.main import RuleKit
from rulekit.rules import SurvivalRule
from tests.utils import assert_rules_are_equals
from tests.utils import dir_path
from tests.utils import get_test_cases


class TestKaplanMeierEstimator(unittest.TestCase):

    survival_rules: survival.SurvivalRules

    def setUp(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]

        self.survival_rules = survival.SurvivalRules(
            survival_time_attr=test_case.survival_time
        )
        example_set = test_case.example_set
        self.survival_rules.fit(
            example_set.values,
            example_set.labels,
        )
        self.km: KaplanMeierEstimator = self.survival_rules.model.rules[
            0
        ].kaplan_meier_estimator

    def test_accessing_probabilities(self):
        self.assertTrue(
            all([p >= 0.0 and p <= 1.0 for p in self.km.probabilities]),
            "All probabilities should be in range [0, 1]",
        )
        self.assertTrue(
            all([isinstance(p, float) for p in self.km.probabilities]),
            "All probabilities should be Pythonic floats",
        )

    def test_accessing_events_count(self):
        self.assertTrue(
            all([isinstance(p, np.int_) for p in self.km.events_count]),
            "All event counts should be Pythonic integers",
        )

    def test_accessing_at_risk_count(self):
        self.assertTrue(
            all([isinstance(p, np.int_) for p in self.km.at_risk_count]),
            "All risk count should be Pythonic integers",
        )


class TestSurvivalRules(unittest.TestCase):

    def test_induction_progress_listener(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]

        surv = survival.SurvivalRules(survival_time_attr=test_case.survival_time)
        example_set = test_case.example_set

        class EventListener(RuleInductionProgressListener):

            lock = threading.Lock()
            induced_rules_count = 0
            on_progress_calls_count = 0

            def on_new_rule(self, rule: SurvivalRule):
                self.lock.acquire()
                self.induced_rules_count += 1
                self.lock.release()

            def on_progress(
                self, total_examples_count: int, uncovered_examples_count: int
            ):
                self.lock.acquire()
                self.on_progress_calls_count += 1
                self.lock.release()

        listener = EventListener()
        surv.add_event_listener(listener)
        surv.fit(
            example_set.values,
            example_set.labels,
        )
        rules_count = len(surv.model.rules)
        self.assertEqual(rules_count, listener.induced_rules_count)
        self.assertEqual(rules_count, listener.on_progress_calls_count)

    def test_compare_with_java_results(self):
        test_cases = get_test_cases("SurvivalLogRankSnCTest")

        for test_case in test_cases:
            params = test_case.induction_params
            tree = survival.SurvivalRules(
                **params, survival_time_attr=test_case.survival_time
            )
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)

    def test_fit_and_predict_on_boolean_columns(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]
        params = test_case.induction_params
        clf = survival.SurvivalRules(
            **params, survival_time_attr=test_case.survival_time
        )
        X, y = test_case.example_set.values, test_case.example_set.labels
        X["boolean_column"] = np.random.randint(low=0, high=2, size=X.shape[0]).astype(
            bool
        )
        clf.fit(X, y)
        clf.predict(X)

        y = pd.Series(y)
        clf.fit(X, y)
        clf.predict(X)

    def test_passing_survival_time_column_to_fit_method(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]
        params = test_case.induction_params
        surv1 = survival.SurvivalRules(**params)
        surv2 = survival.SurvivalRules(
            **params, survival_time_attr=test_case.survival_time
        )
        X, y = test_case.example_set.values, test_case.example_set.labels
        survival_time_col: pd.Series = X[test_case.survival_time]
        X_without_time_col: pd.DataFrame = X.drop(
            columns=[test_case.survival_time], axis=1
        )
        surv1.fit(X_without_time_col, y, survival_time=survival_time_col)
        surv2.fit(X, y)

        assert_rules_are_equals(
            [str(r) for r in surv1.model.rules],
            [str(r) for r in surv2.model.rules],
        )

    def test_ibs_calculation(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]
        params = test_case.induction_params
        surv = survival.SurvivalRules(
            **params, survival_time_attr=test_case.survival_time
        )
        X, y = test_case.example_set.values, test_case.example_set.labels
        survival_time_col: pd.Series = X[test_case.survival_time]
        X_without_time_col: pd.DataFrame = X.drop(
            columns=[test_case.survival_time], axis=1
        )
        surv.fit(X, y)

        ibs: float = surv.score(X, y)
        ibs2: float = surv.score(X_without_time_col, y, survival_time=survival_time_col)

        self.assertEqual(ibs, ibs2)

    def test_getting_training_dataset_kaplan_meier_estimator(self):
        test_case = get_test_cases("SurvivalLogRankSnCTest")[0]
        params = test_case.induction_params
        surv = survival.SurvivalRules(
            **params, survival_time_attr=test_case.survival_time
        )
        example_set = test_case.example_set
        unique_times_counts: int = example_set.values[test_case.survival_time].nunique()
        surv.fit(example_set.values, example_set.labels)

        training_km: KaplanMeierEstimator = surv.get_train_set_kaplan_meier()
        self.assertTrue(
            training_km is not None and isinstance(training_km, KaplanMeierEstimator),
            "Should return KaplanMeierEstimator instance fitted on whole training set",
        )
        self.assertEqual(
            len(training_km.times),
            unique_times_counts,
            "Estimator should contain probabilities for each unique time from the dataset",
        )


class TestExpertSurvivalRules(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases("SurvivalLogRankExpertSnCTest")

        for test_case in test_cases:
            params = test_case.induction_params
            surv = survival.ExpertSurvivalRules(
                **params,
                ignore_missing=True,
                survival_time_attr=test_case.survival_time
            )
            example_set = test_case.example_set
            surv.fit(
                example_set.values,
                example_set.labels,
                expert_rules=test_case.knowledge.expert_rules,
                expert_preferred_conditions=test_case.knowledge.expert_preferred_conditions,
                expert_forbidden_conditions=test_case.knowledge.expert_forbidden_conditions,
            )
            model = surv.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)

    def test_refining_conditions_for_nominal_attributes(self):
        df: pd.DataFrame = read_arff(
            os.path.join(dir_path, "resources", "data", "bmt-train-0.arff")
        )
        X, y = df.drop("survival_status", axis=1), df["survival_status"]

        # Run experiment using python API
        clf = survival.ExpertSurvivalRules(
            complementary_conditions=True,
            extend_using_preferred=False,
            extend_using_automatic=False,
            induce_using_preferred=False,
            induce_using_automatic=False,
            preferred_conditions_per_rule=0,
            preferred_attributes_per_rule=0,
            survival_time_attr="survival_time",
        )
        clf.fit(X, y, expert_rules=[("expert_rules-1", "IF CMVstatus @= {1} THEN")])

        self.assertEqual(
            ["IF [[CMVstatus = {1}]] THEN "],
            [str(r) for r in clf.model.rules],
            "Ruleset should contain only a single rule configured by expert",
        )

        clf.fit(X, y, expert_rules=[("expert_rules-1", "IF CMVstatus @= Any THEN")])
        self.assertEqual(
            ["IF [[CMVstatus = !{1}]] THEN "],
            [str(r) for r in clf.model.rules],
            (
                "Ruleset should contain only a single rule configured by expert with "
                "a refined condition"
            ),
        )

    def test_refining_conditions_for_numerical_attributes(self):
        df: pd.DataFrame = read_arff(
            os.path.join(dir_path, "resources", "data", "bmt-train-0.arff")
        )
        X, y = df.drop("survival_status", axis=1), df["survival_status"]

        # Run experiment using python API
        clf = survival.ExpertSurvivalRules(
            complementary_conditions=True,
            extend_using_preferred=False,
            extend_using_automatic=False,
            induce_using_preferred=False,
            induce_using_automatic=False,
            preferred_conditions_per_rule=0,
            preferred_attributes_per_rule=0,
            survival_time_attr="survival_time",
        )
        clf.fit(X, y, expert_rules=[("expert_rules-1", "IF CD34kgx10d6 @= Any THEN")])
        self.assertEqual(
            ["IF [[CD34kgx10d6 = (-inf, 11.86)]] THEN "],
            [str(r) for r in clf.model.rules],
            (
                "Ruleset should contain only a single rule configured by expert with "
                "a refined condition"
            ),
        )


if __name__ == "__main__":
    unittest.main()
