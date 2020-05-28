from typing import Iterable
import numpy as np

from .helpers import get_rule_generator, create_example_set, PredictionResultMapper


class DecisionTreeRegressor:

    def __init__(self):
        self.rule_generator = get_rule_generator()
        self.model = None
        self._remap_to_numeric = False

    def fit(self, values: Iterable[Iterable], labels: Iterable) -> object:
        example_set = create_example_set(values, labels)
        self.model = self.rule_generator.learn(example_set)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        example_set = create_example_set(values, numeric_labels=True)
        result = self.model.apply(example_set)
        return PredictionResultMapper.map_to_numerical(result)
