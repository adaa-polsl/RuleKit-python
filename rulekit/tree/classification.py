from typing import Iterable, Union, Any, List, Tuple
from numbers import Number
import numpy as np

from .helpers import PredictionResultMapper
from .params import Measures
from .operator import BaseOperator, ExpertKnowledgeOperator


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
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing)
        self._remap_to_numeric = False

    def _map_result(self, predicted_example_set) -> np.ndarray:
        prediction: np.ndarray
        if self._remap_to_numeric:
            prediction = PredictionResultMapper.map_to_numerical(predicted_example_set)
            self._remap_to_numeric = False
        else:
            prediction = PredictionResultMapper.map_to_nominal(predicted_example_set)
        return prediction

    def fit(self, values: Iterable[Iterable], labels: Iterable) -> Any:
        if isinstance(labels[0], Number):
            self._remap_to_numeric = True
            labels = list(map(str, labels))
        super().fit(values, labels)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        return self._map_result(super().predict(values))


class ExpertDecisionTreeClassifier(DecisionTreeClassifier, ExpertKnowledgeOperator):

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
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
        self._remap_to_numeric = False
        ExpertKnowledgeOperator.__init__(
            self,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
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

    def fit(self,
            values: Iterable[Iterable],
            labels: Iterable,
            survival_time_attribute: str = None,

            expert_rules: List[Union[str, Tuple[str, str]]] = None,
            expert_preferred_conditions: List[Union[str, Tuple[str, str]]] = None,
            expert_forbidden_conditions: List[Union[str, Tuple[str, str]]] = None) -> Any:
        if isinstance(labels[0], Number):
            self._remap_to_numeric = True
            labels = list(map(str, labels))
        return ExpertKnowledgeOperator.fit(
            self,
            values,
            labels,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def predict(self, values: Iterable) -> np.ndarray:
        return DecisionTreeClassifier.predict(self, values)
