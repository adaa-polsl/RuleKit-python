import os
import threading
import unittest

import numpy as np
import pandas as pd

from rulekit import regression
from rulekit.arff import read_arff
from rulekit.events import RuleInductionProgressListener
from rulekit.main import RuleKit
from rulekit.params import Measures
from rulekit.rules import RegressionRule
from rulekit.rules import RuleSet
from tests.utils import assert_rules_are_equals
from tests.utils import assert_score_is_greater
from tests.utils import dir_path
from tests.utils import get_test_cases


class TestRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_induction_progress_listener(self):
        test_case = get_test_cases("RegressionSnCTest")[0]

        reg = regression.RuleRegressor()
        example_set = test_case.example_set
        MAX_RULES = 3

        class EventListener(RuleInductionProgressListener):

            lock = threading.Lock()
            induced_rules_count = 0
            on_progress_calls_count = 0

            def on_new_rule(self, rule: RegressionRule):
                self.lock.acquire()
                self.induced_rules_count += 1
                self.lock.release()

            def on_progress(
                self, total_examples_count: int, uncovered_examples_count: int
            ):
                self.lock.acquire()
                self.on_progress_calls_count += 1
                self.lock.release()

            def should_stop(self) -> bool:
                self.lock.acquire()
                should_stop = self.induced_rules_count == MAX_RULES
                self.lock.release()
                return should_stop

        listener = EventListener()
        reg.add_event_listener(listener)
        reg.fit(example_set.values, example_set.labels)

        rules_count = len(reg.model.rules)
        self.assertEqual(rules_count, MAX_RULES)
        self.assertEqual(rules_count, listener.on_progress_calls_count)

    def test_compare_with_java_results(self):
        test_cases = get_test_cases("RegressionSnCTest")

        for test_case in test_cases:
            params = test_case.induction_params
            tree = regression.RuleRegressor(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = [str(r) for r in model.rules]
            assert_rules_are_equals(expected, actual)
            assert_score_is_greater(
                tree.predict(example_set.values), example_set.labels, 0.7
            )

    def test_fit_and_predict_on_boolean_columns(self):
        test_case = get_test_cases("RegressionSnCTest")[0]
        params = test_case.induction_params
        clf = regression.RuleRegressor(**params)
        X, y = test_case.example_set.values, test_case.example_set.labels
        X["boolean_column"] = np.random.randint(low=0, high=2, size=X.shape[0]).astype(
            bool
        )
        clf.fit(X, y)
        clf.predict(X)

        y = pd.Series(y)
        clf.fit(X, y)
        clf.predict(X)

    def test_cholesterol(self):
        resources_dir: str = os.path.join(dir_path, "additional_resources")
        df: pd.DataFrame = read_arff(
            os.path.join(resources_dir, "cholesterol.arff"))
        X, y = df.drop("class", axis=1), df["class"]

        # Run experiment using python API
        reg = regression.RuleRegressor(
            minsupp_new=0.05,
            max_uncovered_fraction=0.0,
            max_growing=0.0,
            induction_measure=Measures.Accuracy,
            pruning_measure=Measures.Accuracy,
            voting_measure=Measures.Accuracy,
            ignore_missing=False,
            select_best_candidate=False,
            complementary_conditions=True,
            max_rule_count=0,
        )
        reg.fit(X, y)
        actual_rules: list[str] = list(map(str, reg.model.rules))
        expected_rules: list[str] = [
            "IF trestbps = (-inf, 149) THEN class = {244.84} [192.73,296.96]",
            "IF trestbps = <122, inf) THEN class = {250.80} [201.79,299.80]",
        ]
        self.assertEqual(actual_rules, expected_rules)


class TestExpertRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases("RegressionExpertSnCTest")

        for test_case in test_cases:
            params = test_case.induction_params
            tree = regression.ExpertRuleRegressor(**params)
            example_set = test_case.example_set
            tree.fit(
                example_set.values,
                example_set.labels,
                expert_rules=test_case.knowledge.expert_rules,
                expert_preferred_conditions=(
                    test_case.knowledge.expert_preferred_conditions
                ),
                expert_forbidden_conditions=(
                    test_case.knowledge.expert_forbidden_conditions
                ),
            )
            model = tree.model
            expected = test_case.reference_report.rules
            actual = [str(r) for r in model.rules]
            assert_rules_are_equals(expected, actual)
            assert_score_is_greater(
                tree.predict(example_set.values), example_set.labels, 0.66
            )

    def test_legacy_expert_rules_format(self):
        """Test if the legacy expert rules format is still supported.
        In legacy format rules strings contains conclusion in the form
        of "label_attr = {NaN}". In new format conclusion part should be empty
        """
        df: pd.DataFrame = read_arff(
            os.path.join(dir_path, "additional_resources", "cholesterol.arff")
        )
        X, y = df.drop("class", axis=1), df["class"]

        # Run experiment using python API
        reg = regression.ExpertRuleRegressor(
            mean_based_regression=True, extend_using_automatic=True
        )
        expert_rule = "IF trestbps < 149 THEN class = {NaN}"
        legacy_ruleset: RuleSet[RegressionRule] = reg.fit(
            X, y, expert_rules=[("0", expert_rule)]
        )
        new_format_ruleset: RuleSet[RegressionRule] = reg.fit(
            X, y, expert_rules=[("rule-0", expert_rule.split("class")[0])]
        )
        legacy_rules: list[str] = list(map(str, legacy_ruleset.rules))
        new_format_rules: list[str] = list(map(str, new_format_ruleset.rules))
        self.assertEqual(legacy_rules, new_format_rules)

    def test_refining_conditions_for_nominal_attributes(self):
        df: pd.DataFrame = read_arff(
            os.path.join(dir_path, "additional_resources", "cholesterol.arff")
        )
        X, y = df.drop("class", axis=1), df["class"]

        # Run experiment using python API
        clf = regression.ExpertRuleRegressor(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
            complementary_conditions=True,
            extend_using_preferred=False,
            extend_using_automatic=False,
            induce_using_preferred=False,
            induce_using_automatic=False,
            preferred_conditions_per_rule=0,
            preferred_attributes_per_rule=0,
        )
        clf.fit(X, y, expert_rules=[("expert_rules-1", "IF sex @= {1} THEN")])
        self.assertEqual(
            ["IF [[sex = {1}]] THEN class = {239.60} [197.06,282.15]"],
            [str(r) for r in clf.model.rules],
            "Ruleset should contain only a single rule configured by expert",
        )

        clf.fit(X, y, expert_rules=[("expert_rules-1", "IF sex @= Any THEN")])
        self.assertEqual(
            ["IF [[sex = {1}]] THEN class = {239.60} [197.06,282.15]"],
            [str(r) for r in clf.model.rules],
            (
                "Ruleset should contain only a single rule configured by expert with "
                "a refined condition"
            ),
        )


if __name__ == "__main__":
    unittest.main()
