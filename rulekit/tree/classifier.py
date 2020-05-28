from typing import Iterable
from numbers import Number
import numpy as np

from .helpers import get_rule_generator, create_example_set, PredictionResultMapper


class DecisionTreeClassifier:

    def __init__(self):
        self.rule_generator = get_rule_generator()
        self.model = None
        self._remap_to_numeric = False

    def _map_result(self, predicted_example_set) -> np.ndarray:
        prediction: np.ndarray
        if self._remap_to_numeric:
            prediction = PredictionResultMapper.map_to_numerical(predicted_example_set)
            self._remap_to_numeric = False
        else:
            prediction = PredictionResultMapper.map_to_nominal(predicted_example_set)
        return prediction

    def fit(self, values: Iterable[Iterable], labels: Iterable) -> object:
        if isinstance(labels[0], Number):
            self._remap_to_numeric = True
            labels = list(map(str, labels))
        example_set = create_example_set(values, labels)
        self.model = self.rule_generator.learn(example_set)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        example_set = create_example_set(values)
        result = self.model.apply(example_set)
        return self._map_result(result)