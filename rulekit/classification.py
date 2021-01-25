from typing import Union, Any, List, Tuple, Dict
from numbers import Number
import numpy as np
import pandas as pd
from enum import Enum
from jpype import JClass

from .helpers import PredictionResultMapper
from .params import Measures
from .operator import BaseOperator, ExpertKnowledgeOperator, Data


class BaseClassifier:

    def __init__(self):
        self.ClassificationRulesPerformance = JClass('adaa.analytics.rules.logic.quality.ClassificationRulesPerformance')

    class MetricTypes(Enum):
        RulesPerExample = 1
        VotingConflicts = 2
        NegativeVotingConflicts = 3

    def _calculate_metric(self, example_set, metric_type) -> float:
        classificationRulesPerformance = self.ClassificationRulesPerformance(metric_type.value)
        classificationRulesPerformance.startCounting(example_set, True)
        return classificationRulesPerformance.getMikroAverage()

    def _calculate_prediction_metrics(self, example_set) -> Dict[str, float]:
        return {
            'rules_per_example': self._calculate_metric(example_set, BaseClassifier.MetricTypes.RulesPerExample),
            'voting_conflicts': self._calculate_metric(example_set, BaseClassifier.MetricTypes.VotingConflicts),
            'negative_voting_conflicts': self._calculate_metric(example_set, BaseClassifier.MetricTypes.NegativeVotingConflicts),
        }


class RuleClassifier(BaseOperator, BaseClassifier):

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        BaseOperator.__init__(
            self,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing)
        BaseClassifier.__init__(self)
        self._remap_to_numeric = False
        self.label_unique_values = []

    def _map_result(self, predicted_example_set) -> np.ndarray:
        prediction: np.ndarray
        if self._remap_to_numeric:
            prediction = PredictionResultMapper.map_to_numerical(predicted_example_set)
            self._remap_to_numeric = False
        else:
            prediction = PredictionResultMapper.map_to_nominal(predicted_example_set)
        return prediction

    def _map_confidence(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map_confidence(predicted_example_set, self.label_unique_values)

    def _get_unique_label_values(self, labels: Data):
        tmp = {}
        for label_value in labels:
            tmp[label_value] = None
        self.label_unique_values = list(tmp.keys())

    def fit(self, values: Data, labels: Data) -> Any:
        self._get_unique_label_values(labels)
        
        if isinstance(labels, pd.DataFrame):
            first_label = labels.iloc[0]
        else:
            first_label = labels[0]
        if isinstance(first_label, Number):
            self._remap_to_numeric = True
            labels = list(map(str, labels))
        BaseOperator.fit(self, values, labels)
        return self

    def predict(self, values: Data, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_result(result_example_set)
        if return_metrics:
            metrics = BaseClassifier._calculate_prediction_metrics(self, result_example_set)
            return (mapped_result_example_set, metrics)
        else:
            return mapped_result_example_set

    def predict_proba(self, values: Data, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_confidence(result_example_set)
        if return_metrics:
            metrics = BaseClassifier._calculate_prediction_metrics(self, result_example_set)
            return (mapped_result_example_set, metrics)
        else:
            return mapped_result_example_set


class ExpertRuleClassifier(RuleClassifier, ExpertKnowledgeOperator, BaseClassifier):

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
            values: Data,
            labels: Data,
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

    def predict(self, values: Data, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        return RuleClassifier.predict(self, values, return_metrics)
