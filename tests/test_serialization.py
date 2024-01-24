import unittest

import sys
import os
import shutil
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{dir_path}/..')

from rulekit.main import RuleKit
from rulekit import classification
import sklearn.tree as scikit
from sklearn.datasets import load_iris, load_diabetes
from sklearn import metrics
import io
import time
import pickle
from rulekit.classification import RuleClassifier, ExpertRuleClassifier
from rulekit.regression import RuleRegressor, ExpertRuleRegressor
from rulekit.survival import SurvivalRules, ExpertSurvivalRules


class TestModelSerialization(unittest.TestCase):

    TMP_DIR_PATH = f'{dir_path}/tmp'
    PICKLE_FILE_PATH = f'{TMP_DIR_PATH}/model.pickle'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TestModelSerialization.TMP_DIR_PATH):
            os.mkdir(TestModelSerialization.TMP_DIR_PATH)
        RuleKit.init()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestModelSerialization.TMP_DIR_PATH)

    def serialize_model(self, model):
        with open(TestModelSerialization.PICKLE_FILE_PATH, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize_model(self) -> object:
        with open(TestModelSerialization.PICKLE_FILE_PATH, 'rb') as handle:
            return pickle.load(handle)

    def test_classifier_serialization(self):
        x, y = load_iris(return_X_y=True)

        model = RuleClassifier(minsupp_new=1)
        model.fit(x, y)
        prediction, metrics = model.predict(x, return_metrics=True)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction, deserialized_model_metrics = deserialized_model.predict(x, return_metrics=True)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')
        self.assertEqual(metrics, deserialized_model_metrics,
                         'Deserialized model should return the same prediction metrics as original one')

        self.serialize_model(deserialized_model)

        deserialized_model = self.deserialize_model()
        deserialized_model_prediction, deserialized_model_metrics = deserialized_model.predict(x, return_metrics=True)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Model deserialized multiple time should predict same as original one')
        self.assertEqual(metrics, deserialized_model_metrics,
                         'Model deserialized multiple time should return the same prediction metrics as original one')


    def test_expert_classifier_serialization(self):
        x, y = load_iris(return_X_y=True)

        model = ExpertRuleClassifier(minsupp_new=1)
        model.fit(x, y)
        prediction = model.predict(x)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction = deserialized_model.predict(x)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')

    def test_regressor_serialization(self):
        x, y = load_diabetes(return_X_y=True)

        model = RuleRegressor(minsupp_new=10)
        model.fit(x, y)
        prediction = model.predict(x)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction = deserialized_model.predict(x)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')


    def test_expert_regressor_serialization(self):
        x, y = load_diabetes(return_X_y=True)

        model = ExpertRuleRegressor(minsupp_new=10)
        model.fit(x, y)
        prediction = model.predict(x)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction = deserialized_model.predict(x)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')

    def test_survival_serialization(self):
        x, y = load_iris(return_X_y=True)

        model = SurvivalRules(minsupp_new=10, survival_time_attr='')
        model.fit(x, y)
        prediction = model.predict(x)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction = deserialized_model.predict(x)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')

    def test_expert_survival_serialization(self):
        x, y = load_iris(return_X_y=True)

        model = ExpertSurvivalRules(minsupp_new=10, survival_time_attr='')
        model.fit(x, y)
        prediction = model.predict(x)

        self.serialize_model(model)
        deserialized_model = self.deserialize_model()
        deserialized_model_prediction = deserialized_model.predict(x)

        self.assertEqual(prediction.all(), deserialized_model_prediction.all(),
                         'Deserialized model should predict same as original one')

    def test_multiple_serialization(self):
        x, y = load_iris(return_X_y=True)

        model = RuleClassifier(minsupp_new=1)
        model.fit(x, y)
        prediction, metrics = model.predict(x, return_metrics=True)

        self.serialize_model(model)
        self.serialize_model(model)
