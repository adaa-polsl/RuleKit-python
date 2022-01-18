from jpype import JClass

from rulekit.params import Measures
from ..rules import Rule, RuleSet
from ..operator import Data
from ..classification import ExpertRuleClassifier
from ..helpers import create_example_set


class ClassificationRuleSetBuilder:

    _java_class_path = 'adaa.analytics.rules.utils.builder.ClassificationRuleSetBuilder'

    def __init__(self) -> None:
        java_class = JClass(ClassificationRuleSetBuilder._java_class_path)
        self._java_object = java_class()

    def add_rule(self, rule: Rule):
        self._java_object.addRule(rule._java_object)
        return self

    def remove_rule(self, index: int):
        self._java_object.removeRule(index)
        return self

    def build(self, values: Data,
              labels: Data,
              induction_measure: Measures = None,
              pruning_measure: Measures = None,
              voting_measure: Measures = None) -> ExpertRuleClassifier:
        clf = ExpertRuleClassifier(
            enable_pruning=False,
            induce_using_automatic=False,
            extend_using_automatic=False,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure
        )
        expert_rules = list(
            map(lambda e: str(e), self._java_object.ruleSet.getRules()))
        clf.fit(values, labels, expert_rules=expert_rules)
        # fix default class prediction and return nan when no rule covers example
        clf.model._java_object.setDefaultClass(-1)
        return clf
