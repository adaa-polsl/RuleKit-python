from typing import Iterable, Any, Union, Tuple, List
import numpy as np
import pandas as pd

from .helpers import PredictionResultMapper
from .operator import BaseOperator, ExpertKnowledgeOperator
from .params import Measures


class SurvivalLogRankTree(BaseOperator):

    def __init__(self,
                 survival_time_attr: str = None,
                 min_rule_covered: int = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        super().__init__(
            min_rule_covered=min_rule_covered,
            induction_measure=None,
            pruning_measure=None,
            voting_measure=None,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing)
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
        return PredictionResultMapper.map(super().predict(values))


class ExpertSurvivalLogRankTree(SurvivalLogRankTree, ExpertKnowledgeOperator):

    def __init__(self,
                 survival_time_attr: str = None,
                 min_rule_covered: int = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None,

                 extend_using_preferred: bool = None,
                 extend_using_automatic: bool = None,
                 induce_using_preferred: bool = None,
                 induce_using_automatic: bool = None,
                 consider_other_classes: bool = None,
                 preferred_conditions_per_rule: int = None,
                 preferred_attributes_per_rule: int = None):
        ExpertKnowledgeOperator.__init__(
            self,
            min_rule_covered=min_rule_covered,
            induction_measure=None,
            pruning_measure=None,
            voting_measure=None,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            extend_using_preferred=extend_using_preferred,
            extend_using_automatic=extend_using_automatic,
            induce_using_preferred=induce_using_preferred,
            induce_using_automatic=induce_using_automatic,
            consider_other_classes=consider_other_classes,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule
        )
        self.survival_time_attr: str = survival_time_attr

    def fit(self,
            values: Iterable[Iterable],
            labels: Iterable,
            survival_time: Iterable = None,

            expert_rules: List[Union[str, Tuple[str, str]]] = None,
            expert_preferred_conditions: List[Union[str, Tuple[str, str]]] = None,
            expert_forbidden_conditions: List[Union[str, Tuple[str, str]]] = None) -> Any:
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError('No "survival_time" attribute name was specified. '
                             'Specify it or pass its values by "survival_time" parameter.')
        if survival_time is not None:
            survival_time_attribute = SurvivalLogRankTree._append_survival_time_columns(values, survival_time)
        else:
            survival_time_attribute = self.survival_time_attr
        return ExpertKnowledgeOperator.fit(
            self,
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def predict(self, values: Iterable) -> np.ndarray:
        return PredictionResultMapper.map(ExpertKnowledgeOperator.predict(self, values))
