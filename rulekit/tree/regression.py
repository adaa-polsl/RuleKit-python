from typing import Iterable, Union, Any
from numbers import Number
import numpy as np

from .operator import BaseOperator
from .params import Measures


class DecisionTreeRegressor(BaseOperator):

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

    def fit(self, values: Iterable[Iterable], labels: Iterable) -> Any:
        if not isinstance(labels[0], Number):
            raise ValueError('DecisionTreeRegressor requires lables values to be numeric')
        super().fit(values, labels)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        return super().predict(values)
