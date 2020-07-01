from typing import Iterable, Any
import numpy as np
import pandas as pd

from .operator import BaseOperator
from .params import Measures


class SurvivalLogRankTree(BaseOperator):

    def __init__(self,
                 survival_time_attr: str = None,
                 min_rule_covered: int = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        super().__init__(
            min_rule_covered,
            Measures.Accuracy,
            Measures.Accuracy,
            Measures.Accuracy,
            max_growing,
            enable_pruning,
            ignore_missing)
        self.survival_time_attr: str = survival_time_attr

    @staticmethod
    def _append_survival_time_columns(values, survival_time) -> str:
        if isinstance(values, pd.Series):
            if survival_time.name is None:
                survival_time.name = 'survival_time'
            values[survival_time.name] = survival_time
            return survival_time.name
        elif isinstance(values, np.ndarray):
            np.append(values, survival_time, axis=1)
        elif isinstance(values, list):
            for index, row in enumerate(values):
                row.append(survival_time[index])
        else:
            raise ValueError('Data values must be instance of either pandas DataFrame, numpy array or list')
        return ''

    def fit(self, values: Iterable[Iterable], labels: Iterable, survival_time: Iterable = None) -> Any:
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError('No "survival_time" attribute name was specified. '
                             'Specify it or pass its values by "survival_time" parameter.')
        if survival_time is not None:
            survival_time_attribute = SurvivalLogRankTree._append_survival_time_columns(values, survival_time)
        else:
            survival_time_attribute = self.survival_time_attr
        super().fit(values, labels, survival_time_attribute)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        return super().predict(values)
