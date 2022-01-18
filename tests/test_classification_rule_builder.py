import unittest
from typing import List

from jpype import JClass
from rulekit import RuleKit
from rulekit.rules import Rule
from rulekit.builder.rules import ClassificationRuleBuilder
from sklearn.datasets import load_iris
from rulekit import classification


class TestClassificationRuleBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    @staticmethod
    def get_ref_rules() -> List[Rule]:
        clf = classification.RuleClassifier()
        X, y = load_iris(return_X_y=True)
        clf.fit(X, y)
        return clf.model.rules

    def test_building_from_existing_rule(self):
        ref_rules: List[Rule] = TestClassificationRuleBuilder.get_ref_rules()
        ref_rule: Rule = ref_rules[0]

        builder = ClassificationRuleBuilder.from_rule(ref_rule, 'label')

        rule: Rule = builder.build()
        self.assertEqual(str(ref_rule), str(rule), 'Ref rule and built one should be the same')

    def test_building_from_scratch(self):
        ref_rules: List[Rule] = TestClassificationRuleBuilder.get_ref_rules()
        ref_rule: Rule = ref_rules[0]
        
        builder = ClassificationRuleBuilder('label')
        rule: Rule = builder.setPremise(ref_rule.premise).setConsequence(1.0).build()

        self.assertEqual(ref_rule.premise, rule.premise, 'Ref rule premise and built rule premise should be the same')
        self.assertEqual(str(rule.consequence), 'label = {1}', 'Ref rule consequence and built rule consequence should be the same')


if __name__ == '__main__':
    unittest.main()
