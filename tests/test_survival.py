import unittest
import threading

from rulekit.main import RuleKit
from rulekit import survival
from rulekit.rules import Rule
from rulekit.events import RuleInductionProgressListener
from tests.utils import get_test_cases, assert_rules_are_equals


class TestSurvivalLogRankTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_induction_progress_listener(self):
        test_case = get_test_cases('SurvivalLogRankSnCTest')[0]

        surv = survival.SurvivalRules(survival_time_attr=test_case.survival_time)
        example_set = test_case.example_set

        class EventListener(RuleInductionProgressListener):

            lock = threading.Lock()
            induced_rules_count = 0
            on_progress_calls_count = 0
            should_stop_calls_count = 0

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

            def should_stop(self) -> bool:
                self.lock.acquire()
                self.should_stop_calls_count += 1
                self.lock.release()
                return False

        listener = EventListener()
        surv.add_event_listener(listener)
        surv.fit(
            example_set.values,
            example_set.labels,
        )
        rules_count = len(surv.model.rules)
        self.assertEqual(rules_count, listener.induced_rules_count)
        self.assertEqual(rules_count, listener.on_progress_calls_count)
        self.assertEqual(rules_count, listener.should_stop_calls_count)

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
            actual = list(map(lambda e: str(e), model.rules))
            assert_rules_are_equals(expected, actual)


if __name__ == '__main__':
    unittest.main()
