import unittest

from rulekit.main import RuleKit
import rulekit.tree.classifier as rulekit
import sklearn.tree as scikit
from sklearn.datasets import load_iris
from sklearn import metrics


class TestDecisionTreeClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        RuleKit.init()

    def test_classification_accuracy_on_iris(self):
        scikit_clf = scikit.DecisionTreeClassifier()
        rulekit_clf = rulekit.DecisionTreeClassifier()
        x, y = load_iris(return_X_y=True)

        scikit_clf.fit(x, y)
        rulekit_clf.fit(x, y)
        scikit_prediction = scikit_clf.predict(x)
        rulekit_prediction = rulekit_clf.predict(x)

        scikit_accuracy = metrics.accuracy_score(y, scikit_prediction)
        rulekit_accuracy = metrics.accuracy_score(y, rulekit_prediction)

        assert(abs(scikit_accuracy - rulekit_accuracy) < 0.03, 'RuleKit model should perform similar to scikit model')


if __name__ == '__main__':
    unittest.main()
