import unittest

from rulekit.main import RuleKit
from rulekit import classification
import sklearn.tree as scikit
from sklearn.datasets import load_iris
from sklearn import metrics

from tests.utils import get_test_cases, assert_rules_are_equals, assert_accuracy_is_greater


class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_classification_accuracy_on_iris(self):
        scikit_clf = scikit.DecisionTreeClassifier()
        rulekit_clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        scikit_clf.fit(x, y)
        rulekit_clf.fit(x, y)
        scikit_prediction = scikit_clf.predict(x)
        rulekit_prediction = rulekit_clf.predict(x)
        scikit_accuracy = metrics.accuracy_score(y, scikit_prediction)
        rulekit_accuracy = metrics.accuracy_score(y, rulekit_prediction)

        assert abs(scikit_accuracy - rulekit_accuracy) < 0.03, 'RuleKit model should perform similar to scikit model'

    def test_getting_examples_coverage(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)

        coverage_matrix = clf.get_coverage_matrix(x)
        num_rows, num_cols = coverage_matrix.shape

        self.assertEqual(num_rows, len(x), 'Coverage matrix should have as many rows as examples in dataset')
        self.assertEqual(num_cols, len(clf.model.rules), 'Coverage matrix should have as many cols as rules in ruleset')

    def test_classification_metrics(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)
        rulekit_prediction, m = clf.predict(x, return_metrics=True)
        self.assertIsNotNone(m['rules_per_example'], 'rules_per_example should be calculated')
        self.assertIsNotNone(m['voting_conflicts'], 'rules_per_example should be calculated')
        self.assertIsNotNone(m['negative_voting_conflicts'], 'rules_per_example should be calculated')

    def test_classification_predict_proba(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)
        confidence_matrix, m = clf.predict_proba(x, return_metrics=True)
        for row in confidence_matrix:
            sum = 0
            for col in row:
                sum += col
            self.assertAlmostEquals(sum, 1, 3, 'Confidence matrix rows should sum to 1')

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('ClassificationSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = classification.RuleClassifier(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)
            assert_accuracy_is_greater(tree.predict(example_set.values), example_set.labels, 0.9)


class TestExpertDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('ClassificationExpertSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = classification.ExpertRuleClassifier(**params)
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
            assert_accuracy_is_greater(tree.predict(example_set.values), example_set.labels, 0.9)


if __name__ == '__main__':
    unittest.main()
