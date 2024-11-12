import threading
import unittest

import numpy as np
import pandas as pd

from rulekit import survival
from rulekit.events import RuleInductionProgressListener
from rulekit.kaplan_meier import KaplanMeierEstimator
from rulekit.main import RuleKit
from rulekit.rules import SurvivalRule
from tests.utils import assert_rules_are_equals
from tests.utils import get_test_cases


class TestKaplanMeierEstimator(unittest.TestCase):

    survival_rules: survival.SurvivalRules

    def setUp(self):
        test_case = get_test_cases('SurvivalLogRankSnCTest')[0]

        self.survival_rules = survival.SurvivalRules(
            survival_time_attr=test_case.survival_time)
        example_set = test_case.example_set
        self.survival_rules.fit(
            example_set.values,
            example_set.labels,
        )
        self.km: KaplanMeierEstimator = self.survival_rules.model.rules[0].kaplan_meier_estimator

    def test_accessing_probabilities(self):
        self.assertTrue(all([
            p >= 0.0 and p <= 1.0
            for p in self.km.probabilities
        ]), 'All probabilities should be in range [0, 1]')
        self.assertTrue(all([
            isinstance(p, float)
            for p in self.km.probabilities
        ]), 'All probabilities should be Pythonic floats')

    def test_accessing_events_count(self):
        self.assertTrue(all([
            isinstance(p, np.int_)
            for p in self.km.events_count
        ]), 'All event counts should be Pythonic integers')

    def test_accessing_at_risk_count(self):
        self.assertTrue(all([
            isinstance(p, np.int_)
            for p in self.km.at_risk_count
        ]), 'All risk count should be Pythonic integers')


class TestSurvivalRules(unittest.TestCase):

    def test_induction_progress_listener(self):
        test_case = get_test_cases('SurvivalLogRankSnCTest')[0]

        surv = survival.SurvivalRules(
            survival_time_attr=test_case.survival_time)
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
                self,
                total_examples_count: int,
                uncovered_examples_count: int
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
        test_cases = get_test_cases('SurvivalLogRankSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = survival.SurvivalRules(
                **params, survival_time_attr=test_case.survival_time)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)

    def test_fit_and_predict_on_boolean_columns(self):
        test_case = get_test_cases('SurvivalLogRankSnCTest')[0]
        params = test_case.induction_params
        clf = survival.SurvivalRules(
            **params, survival_time_attr=test_case.survival_time
        )
        X, y = test_case.example_set.values, test_case.example_set.labels
        X['boolean_column'] = np.random.randint(
            low=0, high=2, size=X.shape[0]).astype(bool)
        clf.fit(X, y)
        clf.predict(X)

        y = pd.Series(y)
        clf.fit(X, y)
        p = clf.predict(X)


class TestExpertSurvivalLogRankTree(unittest.TestCase):

    @ classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('SurvivalLogRankExpertSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = survival.ExpertSurvivalRules(
                **params, ignore_missing=True, survival_time_attr=test_case.survival_time)
            example_set = test_case.example_set
            tree.fit(example_set.values,
                     example_set.labels,
                     expert_rules=test_case.knowledge.expert_rules,
                     expert_preferred_conditions=test_case.knowledge.expert_preferred_conditions,
                     expert_forbidden_conditions=test_case.knowledge.expert_forbidden_conditions)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)


if __name__ == '__main__':
    unittest.main()
