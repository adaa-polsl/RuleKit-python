import unittest

from rulekit.main import RuleKit
from rulekit import regression

from .utils import get_test_cases, assert_rules_are_equals


class TestDecisionTreeRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('RegressionSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = regression.DecisionTreeRegressor(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)


if __name__ == '__main__':
    unittest.main()
