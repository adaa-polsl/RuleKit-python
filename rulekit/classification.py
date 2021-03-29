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
    """:meta private:"""

    def __init__(self):
        self.ClassificationRulesPerformance = JClass('adaa.analytics.rules.logic.quality.ClassificationRulesPerformance')

    class MetricTypes(Enum):
        """:meta private:"""
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
    """Classification model."""

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        """
        Parameters
        ----------
        min_rule_covered : int = None
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule 
            (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = None
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = None
            measure used during pruning. Could be user defined (string), for example  :code:`2 * p / n`; 
            default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = None
            measure used during voting; default measure is correlation
        max_growing : int = None
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase 
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = None
            enable or disable pruning, default is True.
        ignore_missing : bool = None
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always 
            considered as not fulfilling the condition build upon that attribute); default: False.
        """
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
        if len(self.label_unique_values) > 0 and isinstance(self.label_unique_values[0], bytes):
            self.label_unique_values = [item.decode('utf-8') for item in self.label_unique_values]

    def fit(self, values: Data, labels: Data) -> Any:
        """Train model on given dataset.
    
        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            labels
        Returns
        -------
        self : RuleClassifier
        """
        self._get_unique_label_values(labels)
        
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            first_label = labels.iloc[0]
        else:
            first_label = labels[0]
        if isinstance(first_label, Number):
            self._remap_to_numeric = True
            labels = list(map(str, labels))
        BaseOperator.fit(self, values, labels)
        return self

    def predict(self, values: Data, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Perform prediction and returns predicted labels.
    
        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        return_metrics: bool = False
            Optional flag. If set to *True* method will calculate some additional model metrics. 
            Method will then return tuple instead of just predicted labels.
 
        Returns
        -------
        result : Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just prediction, otherwise a tuple will be returned with first
            element being prediction and second one being metrics.
        """
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_result(result_example_set)
        if return_metrics:
            metrics = BaseClassifier._calculate_prediction_metrics(self, result_example_set)
            return (mapped_result_example_set, metrics)
        else:
            return mapped_result_example_set

    def predict_proba(self, values: Data, return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Perform prediction and returns class probabilities for each example.
    
        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        return_metrics: bool = False
            Optional flag. If set to *True* method will calculate some additional model metrics. 
            Method will then return tuple instead of just probabilities.
 
        Returns
        -------
        result : Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just probabilities matrix, otherwise a tuple will be returned with first
            element being prediction and second one being metrics.
        """
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_confidence(result_example_set)
        if return_metrics:
            metrics = BaseClassifier._calculate_prediction_metrics(self, result_example_set)
            return (mapped_result_example_set, metrics)
        else:
            return mapped_result_example_set


class ExpertRuleClassifier(RuleClassifier, ExpertKnowledgeOperator, BaseClassifier):
    """Classification model using expert knowledge."""

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
        """
        Parameters
        ----------
        min_rule_covered : int = None
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule 
            (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = None
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = None
            measure used during pruning. Could be user defined (string), for example  :code:`2 * p / n`; 
            default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = None
            measure used during voting; default measure is correlation
        max_growing : int = None
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase 
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = None
            enable or disable pruning, default is True.
        ignore_missing : bool = None
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always 
            considered as not fulfilling the condition build upon that attribute); default: False.

        extend_using_preferred : bool = None
            boolean indicating whether initial rules should be extended with a use of preferred conditions and attributes; default is False
        extend_using_automatic : bool = None
            boolean indicating whether initial rules should be extended with a use of automatic conditions and attributes; default is False
        induce_using_preferred : bool = None
            boolean indicating whether new rules should be induced with a use of preferred conditions and attributes; default is False
        induce_using_automatic : bool = None
            boolean indicating whether new rules should be induced with a use of automatic conditions and attributes; default is False
        consider_other_classes : bool = None
            boolean indicating whether automatic induction should be performed for classes for which no user's knowledge has been defined (classification only); default is False.
        preferred_conditions_per_rule : int = None
            maximum number of preferred conditions per rule,
        preferred_attributes_per_rule : int = None
            maximum number of preferred attributes per rule,
        """
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

            expert_rules: List[Union[str, Tuple[str, str]]] = None,
            expert_preferred_conditions: List[Union[str, Tuple[str, str]]] = None,
            expert_forbidden_conditions: List[Union[str, Tuple[str, str]]] = None) -> Any:
        """Train model on given dataset.
    
        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            labels
        
        expert_rules : List[Union[str, Tuple[str, str]]]
             set of initial rules, either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        expert_preferred_conditions : List[Union[str, Tuple[str, str]]]
             multiset of preferred conditions (used also for specifying preferred attributes by using special value Any). Either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        expert_forbidden_conditions : List[Union[str, Tuple[str, str]]]
             set of forbidden conditions (used also for specifying forbidden attributes by using special valye Any). Either passed as a list of strings representing rules or as list of tuples where first
             element is name of the rule and second one is rule string.
        Returns
        -------
        self : ExpertRuleClassifier
        """
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