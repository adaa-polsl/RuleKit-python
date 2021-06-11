import unittest

from rulekit.main import RuleKit
from rulekit import regression

from tests.utils import get_test_cases, assert_rules_are_equals, assert_score_is_greater


class TestRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('RegressionSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = regression.RuleRegressor(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)
            assert_score_is_greater(tree.predict(example_set.values), example_set.labels, 0.7)


class TestExpertRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('RegressionExpertSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = regression.ExpertRuleRegressor(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values,
                     example_set.labels,
                     expert_rules=test_case.knowledge.expert_rules,
                     expert_preferred_conditions=test_case.knowledge.expert_preferred_conditions,
                     expert_forbidden_conditions=test_case.knowledge.expert_forbidden_conditions)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)
            assert_score_is_greater(tree.predict(example_set.values), example_set.labels, 0.66)


if __name__ == '__main__':
    unittest.main()
