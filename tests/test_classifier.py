import os
import threading
import unittest

import numpy as np
import pandas as pd
import sklearn.tree as scikit
from scipy.io import arff
from sklearn import metrics
from sklearn.datasets import load_iris

from rulekit import classification
from rulekit.events import RuleInductionProgressListener
from rulekit.params import Measures
from rulekit.rules import Rule
from tests.utils import assert_accuracy_is_greater
from tests.utils import assert_rules_are_equals
from tests.utils import dir_path
from tests.utils import get_test_cases


class TestClassifier(unittest.TestCase):

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

        assert abs(scikit_accuracy -
                   rulekit_accuracy) < 0.04, 'RuleKit model should perform similar to scikit model'

    def test_induction_progress_listener(self):
        rulekit_clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        class EventListener(RuleInductionProgressListener):

            lock = threading.Lock()
            induced_rules_count = 0
            on_progress_calls_count = 0

            def on_new_rule(self, rule: Rule):
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
        rulekit_clf.add_event_listener(listener)
        rulekit_clf.fit(x, y)
        rules_count = len(rulekit_clf.model.rules)
        self.assertEqual(
            rules_count, listener.induced_rules_count
        )
        self.assertEqual(rules_count, listener.on_progress_calls_count)

    def test_getting_examples_coverage(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)

        coverage_matrix = clf.get_coverage_matrix(x)
        num_rows, num_cols = coverage_matrix.shape

        self.assertEqual(num_rows, len(
            x), 'Coverage matrix should have as many rows as examples in dataset')
        self.assertEqual(num_cols, len(
            clf.model.rules), 'Coverage matrix should have as many cols as rules in ruleset')

    def test_classification_metrics(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)
        y_pred, m = clf.predict(x, return_metrics=True)
        self.assertEqual(len(y_pred), len(y))
        self.assertIsNotNone(m['rules_per_example'],
                             'rules_per_example should be calculated')
        self.assertIsNotNone(m['voting_conflicts'],
                             'rules_per_example should be calculated')

    def test_classification_predict_proba(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)
        confidence_matrix, m = clf.predict_proba(x, return_metrics=True)
        for row in confidence_matrix:
            sum = 0
            for col in row:
                sum += col
            self.assertAlmostEqual(
                sum, 1, 3, 'Confidence matrix rows should sum to 1')

    def test_prediction_results_mapping(self):
        """
        This method tests classifications on numeric labels which possible values does
        not start from 0. RuleKit undehood maps all labels values to integer values starting
        from 0 to N (counting by order of appearance in dataset). Those maped values must be
        later remaped back to actual label value. This test verifies that predict method returns
        correct (remaped) label value.
        """
        clf = classification.RuleClassifier()

        # some trivial dataset - OR (2 = false, 3 = true)
        x = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        y = np.array([0., 1., 0., 0.])
        clf.fit(x, y)
        prediction = clf.predict(x)

        self.assertEqual(y.all(), prediction.all())

    def test_prediction_on_nominal_values(self):
        clf = classification.RuleClassifier()

        # some trivial dataset - AND Gate
        x = np.array([[0, 1], [1, 1], [1, 0], [0, 0]])
        y = np.array(['false', 'true', 'false', 'false'])
        clf.fit(x, y)
        prediction = clf.predict(x)

        self.assertTrue(np.array_equal(y, prediction))

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('ClassificationSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            tree = classification.RuleClassifier(**params)
            example_set = test_case.example_set
            tree.fit(example_set.values, example_set.labels)
            model = tree.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)
            assert_accuracy_is_greater(tree.predict(
                example_set.values), example_set.labels, 0.9)

    def test_predict_proba(self):
        test_case = get_test_cases('ClassificationSnCTest')[0]
        params = test_case.induction_params
        clf = classification.RuleClassifier(**params)
        example_set = test_case.example_set
        clf.fit(
            example_set.values,
            example_set.labels,
        )
        res = clf.predict_proba(example_set.values)
        self.assertEqual(
            res.shape[0],
            example_set.values.shape[0],
            'Should have as many rows as the original dataset'
        )
        self.assertEqual(
            res.shape[1],
            np.unique(example_set.labels).shape[0],
            'Should have as many columns as there are classes in the dataset'
        )
        self.assertTrue(
            res.max() <= 1 and res.min() >= 0,
            'Predicted probabilities should be in range [0, 1]'
        )

    def test_fit_and_predict_on_boolean_columns(self):
        test_case = get_test_cases('ClassificationSnCTest')[0]
        params = test_case.induction_params
        clf = classification.RuleClassifier(**params)
        X, y = test_case.example_set.values, test_case.example_set.labels
        X['boolean_column'] = np.random.randint(
            low=0, high=2, size=X.shape[0]).astype(bool)
        clf.fit(X, y)
        clf.predict(X)

        y = y.astype(bool)
        clf.fit(X, y)
        clf.predict(X)

        y = pd.Series(y)
        clf.fit(X, y)
        clf.predict(X)


class TestExperClassifier(unittest.TestCase):

    def test_compare_with_java_results(self):
        test_cases = get_test_cases('ClassificationExpertSnCTest')

        for test_case in test_cases:
            params = test_case.induction_params
            clf = classification.ExpertRuleClassifier(**params)
            example_set = test_case.example_set
            clf.fit(example_set.values,
                    example_set.labels,
                    expert_rules=test_case.knowledge.expert_rules,
                    expert_preferred_conditions=test_case.knowledge.expert_preferred_conditions,
                    expert_forbidden_conditions=test_case.knowledge.expert_forbidden_conditions)
            model = clf.model
            expected = test_case.reference_report.rules
            actual = list(map(str, model.rules))
            assert_rules_are_equals(expected, actual)
            assert_accuracy_is_greater(clf.predict(
                example_set.values), example_set.labels, 0.75)

    def test_predict_proba(self):
        test_case = get_test_cases('ClassificationExpertSnCTest')[0]
        params = test_case.induction_params
        clf = classification.ExpertRuleClassifier(**params)
        example_set = test_case.example_set
        clf.fit(example_set.values,
                example_set.labels,
                expert_rules=test_case.knowledge.expert_rules,
                expert_preferred_conditions=test_case.knowledge.expert_preferred_conditions,
                expert_forbidden_conditions=test_case.knowledge.expert_forbidden_conditions)
        res = clf.predict_proba(example_set.values)
        self.assertEqual(
            res.shape[0],
            example_set.values.shape[0],
            'Should have as many rows as the original dataset'
        )
        self.assertEqual(
            res.shape[1],
            np.unique(example_set.labels).shape[0],
            'Should have as many columns as there are classes in the dataset'
        )
        self.assertTrue(
            res.max() <= 1 and res.min() >= 0,
            'Predicted probabilities should be in range [0, 1]'
        )

    # Issue #17
    def test_left_open_intervals_in_expert_induction(self):
        df = pd.DataFrame(arff.loadarff(
            f'{dir_path}/resources/data/seismic-bumps-train-minimal.arff')[0]
        )
        X = df.drop('class', axis=1)
        y = df['class']

        expert_rules = [
            ('rule-0', 'IF [[gimpuls = <-inf, 750)]] THEN class = {0}'),
            ('rule-1', 'IF [[gimpuls = (750, inf)]] THEN class = {1}')
        ]

        expert_preferred_conditions = [
            ('preferred-condition-0',
             '1: IF [[seismic = {a}]] THEN class = {0}'),
            ('preferred-attribute-0',
             '1: IF [[gimpuls = Any]] THEN class = {1}')
        ]

        expert_forbidden_conditions = [
            ('forb-attribute-0',
             '1: IF [[seismoacoustic  = Any]] THEN class = {0}'),
            ('forb-attribute-1', 'inf: IF [[ghazard  = Any]] THEN class = {1}')
        ]
        clf = classification.ExpertRuleClassifier(
            minsupp_new=8,
            max_growing=0,
            extend_using_preferred=True,
            extend_using_automatic=True,
            induce_using_preferred=True,
            induce_using_automatic=True
        )
        clf.fit(
            X, y,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def test_user_defined_measures(self):
        def full_coverage(p: float, n: float, P: float, N: float) -> float:
            return (p + n) / (P + N)

        python_clf = classification.RuleClassifier(
            induction_measure=full_coverage,
            pruning_measure=full_coverage,
            voting_measure=full_coverage,
        )
        java_clf = classification.RuleClassifier(
            induction_measure=Measures.FullCoverage,
            pruning_measure=Measures.FullCoverage,
            voting_measure=Measures.FullCoverage,
        )
        x, y = load_iris(return_X_y=True)

        python_clf.fit(x, y)
        java_clf.fit(x, y)

        self.assertEqual(
            [r.weight for r in python_clf.model.rules],
            [r.weight for r in java_clf.model.rules],
            'Weights should be equal'
        )
        self.assertEqual(
            [str(r) for r in python_clf.model.rules],
            [str(r) for r in java_clf.model.rules],
            'Rules should be equal'
        )

        def zero_measure(p: float, n: float, P: float, N: float) -> float:
            return 0.0

        python_clf2 = classification.RuleClassifier(
            induction_measure=Measures.FullCoverage,
            pruning_measure=Measures.FullCoverage,
            voting_measure=zero_measure,
        )
        python_clf2.fit(x, y)
        self.assertTrue(all([
            r.weight == 0.0 for r in python_clf2.model.rules
        ]))


if __name__ == '__main__':
    unittest.main()
