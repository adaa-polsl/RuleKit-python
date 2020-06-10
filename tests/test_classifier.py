import unittest

from rulekit.main import RuleKit
import rulekit.tree.classifier as rulekit
import sklearn.tree as scikit
from sklearn.datasets import load_iris
from sklearn import metrics
from scipy.io import arff
import pandas as pd

from .utils import get_dataset_path


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

    def test_classification_rules_on_deals(self):
        train_data = arff.loadarff(get_dataset_path('deals/deals-train.arff'))
        test_data = arff.loadarff(get_dataset_path('deals/deals-test.arff'))
        train_df = pd.DataFrame(train_data[0])
        train_x = train_df[['Age', 'Gender', 'Payment Method']]
        train_y = train_df['Future Customer']
        test_df = pd.DataFrame(test_data[0])
        test_x = test_df[['Age', 'Gender', 'Payment Method']]
        test_y = test_df['Future Customer'].to_numpy(dtype=str)

        tree = rulekit.DecisionTreeClassifier()
        tree.fit(train_x, train_y)
        prediction = tree.predict(test_x)

        accuracy = metrics.accuracy_score(test_y, prediction)
        print(accuracy)


if __name__ == '__main__':
    unittest.main()
