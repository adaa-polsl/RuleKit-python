from typing import Iterable, Union
from numbers import Number
import numpy as np

from .helpers import PredictionResultMapper
from .params import Measures
from .operator import BaseOperator


class DecisionTreeClassifier(BaseOperator):

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        super().__init__(
            min_rule_covered,
            induction_measure,
            pruning_measure,
            voting_measure,
            max_growing,
            enable_pruning,
            ignore_missing)
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
        super().fit(values, labels)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        return self._map_result(super().predict(values))