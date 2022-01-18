import unittest
from typing import List

from jpype import JClass
from rulekit import RuleKit
from rulekit.params import Measures
from rulekit.rules import Rule, RuleSet
from rulekit.builder.ruleset import ClassificationRuleSetBuilder
from sklearn.datasets import load_iris
from rulekit import classification


class TestClassificationRuleSetBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    @staticmethod
    def get_ref_rules() -> List[Rule]:
        clf = classification.RuleClassifier(induction_measure=Measures.Accuracy,
                                            pruning_measure=Measures.Accuracy)
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        return clf.model.rules

    def test_adding_rules(self):
        X, y = load_iris(return_X_y=True)
        ref_rules: List[Rule] = TestClassificationRuleSetBuilder.get_ref_rules()

        builder = ClassificationRuleSetBuilder()
        model: classification.ExpertRuleClassifier = builder.add_rule(ref_rules[0]).build(X, y)

        tmp = str(model.model.rules[0]).replace('[', '')
        new_rule = tmp.replace(']', '')
        self.assertEqual(
            new_rule, str(ref_rules[0]), 'Rules should be the same')
        self.assertEqual(len(model.model.rules), 1,
                         'Ruleset should contain single rule')


if __name__ == '__main__':
    unittest.main()
