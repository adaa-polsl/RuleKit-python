import unittest

from rulekit.main import RuleKit
from rulekit import survival

from tests.utils import get_test_cases, assert_rules_are_equals


class TestSurvivalLogRankTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('SurvivalLogRankSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = survival.SurvivalRules(**params, survival_time_attr=test_case.survival_time)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)


class TestExpertSurvivalLogRankTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('SurvivalLogRankExpertSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = survival.ExpertSurvivalRules(**params, ignore_missing=True, survival_time_attr=test_case.survival_time)
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


if __name__ == '__main__':
    unittest.main()
