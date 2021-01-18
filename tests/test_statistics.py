import unittest

from rulekit.main import RuleKit
from rulekit import classification
from rulekit.rules import RuleSetStatistics
from sklearn.datasets import load_iris


class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_classification_accuracy_on_iris(self):
        clf = classification.RuleClassifier()
        x, y = load_iris(return_X_y=True)

        clf.fit(x, y)

        ruleset_stats: RuleSetStatistics = clf.model.stats
        for rule in clf.model.rules:
            rule.stats


if __name__ == '__main__':
    unittest.main()
