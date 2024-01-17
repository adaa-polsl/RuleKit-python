"""Module contaiing classes for classification analysis and prediction.
"""
from __future__ import annotations
from numbers import Number
from typing import Union, Optional, Iterable
from enum import Enum
import numpy as np
import pandas as pd
from sklearn import metrics
from jpype import JClass

from ._helpers import PredictionResultMapper
from .params import (
    Measures,
    ModelsParams,
    DEFAULT_PARAMS_VALUE,
    ContrastSetModelParams
)
from ._operator import (
    Data,
    BaseOperator,
    ExpertKnowledgeOperator,
)


class ClassificationParams(ModelsParams):
    control_apriori_precision: bool = DEFAULT_PARAMS_VALUE['control_apriori_precision']
    approximate_induction: bool = DEFAULT_PARAMS_VALUE['approximate_induction']
    approximate_bins_count: int = DEFAULT_PARAMS_VALUE['approximate_bins_count']


class BaseClassifier:
    """:meta private:"""

    def __init__(self):
        self._init_classification_rule_performance()

    class MetricTypes(Enum):
        """:meta private:"""
        RulesPerExample = 1  # pylint: disable=invalid-name
        VotingConflicts = 2  # pylint: disable=invalid-name
        NegativeVotingConflicts = 3  # pylint: disable=invalid-name

    def _init_classification_rule_performance(self):
        self.ClassificationRulesPerformance = JClass(  # pylint: disable=invalid-name
            'adaa.analytics.rules.logic.quality.ClassificationRulesPerformance'
        )

    def _calculate_metric(self, example_set, metric_type) -> float:
        classification_rules_performance = self.ClassificationRulesPerformance(
            metric_type.value)
        classification_rules_performance.startCounting(example_set, True)
        return classification_rules_performance.getMikroAverage()

    def _calculate_prediction_metrics(self, example_set) -> dict[str, float]:
        return {
            'rules_per_example': self._calculate_metric(
                example_set, BaseClassifier.MetricTypes.RulesPerExample
            ),
            'voting_conflicts': self._calculate_metric(
                example_set, BaseClassifier.MetricTypes.VotingConflicts
            ),
            'negative_voting_conflicts': self._calculate_metric(
                example_set, BaseClassifier.MetricTypes.NegativeVotingConflicts
            ),
        }


class RuleClassifier(BaseOperator, BaseClassifier):
    """Classification model."""

    __params_class__ = ClassificationParams

    def __init__(
        self,
        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE['induction_measure'],
        pruning_measure: Union[Measures,
                               str] = DEFAULT_PARAMS_VALUE['pruning_measure'],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE['voting_measure'],
        max_growing: float = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],
        control_apriori_precision: bool = DEFAULT_PARAMS_VALUE['control_apriori_precision'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
        approximate_induction: bool = DEFAULT_PARAMS_VALUE['approximate_induction'],
        approximate_bins_count: int = DEFAULT_PARAMS_VALUE['approximate_bins_count'],
        min_rule_covered: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be
            covered by a new rule (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = :class:`rulekit.params.\
            Measures.Correlation`
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during pruning. Could be user defined (string), for example  
            :code:`2 * p / n`; default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during voting; default measure is correlation
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added
            to the rule in the growing phase (use this parameter for large datasets if execution
            time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value
            of given attribute is always cconsidered as not fulfilling the condition build upon 
            that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default:
            False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        control_apriori_precision : bool = True
            When inducing classification rules, verify if candidate precision is higher than 
            apriori precision of the investigated class.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.
        approximate_induction: bool = False
            Use an approximate induction heuristic which does not check all possible splits;
            note: this is an experimental feature and currently works only for classification
            data sets, results may change in future;
        approximate_bins_count: int = 100
            maximum number of bins for an attribute evaluated in the approximate induction.
        min_rule_covered : int = None
            alias to `minsupp_new`. Parameter is deprecated and will be removed in the next major
            version, use `minsupp_new`

            .. deprecated:: 1.7.0
                Use parameter `minsupp_new` instead.
        """
        BaseOperator.__init__(
            self,
            minsupp_new=minsupp_new,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            control_apriori_precision=control_apriori_precision,
            max_rule_count=max_rule_count,
            approximate_induction=approximate_induction,
            approximate_bins_count=approximate_bins_count,
        )
        BaseClassifier.__init__(self)
        self._remap_to_numeric = False
        self.label_unique_values = []

    def _map_result(self, predicted_example_set) -> np.ndarray:
        prediction: np.ndarray
        if self._remap_to_numeric:
            prediction = PredictionResultMapper.map_to_numerical(
                predicted_example_set)
        else:
            prediction = PredictionResultMapper.map_to_nominal(
                predicted_example_set)
        return prediction

    def _map_confidence(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map_confidence(
            predicted_example_set, self.label_unique_values
        )

    def _get_unique_label_values(self, labels: Data):
        tmp = {}
        for label_value in labels:
            tmp[label_value] = None
        self.label_unique_values = list(tmp.keys())
        if len(self.label_unique_values) > 0 and isinstance(self.label_unique_values[0], bytes):
            self.label_unique_values = [item.decode(
                'utf-8') for item in self.label_unique_values]

    def _prepare_labels(self, labels: Data) -> Data:
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            if isinstance(labels.iloc[0], Number):
                self._remap_to_numeric = True
                return labels.astype(str)
        else:
            if isinstance(labels[0], Number):
                self._remap_to_numeric = True
                return list(map(str, labels))
        return labels

    def fit(self, values: Data, labels: Data) -> RuleClassifier:  # pylint: disable=arguments-differ
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
        labels = self._prepare_labels(labels)
        BaseOperator.fit(self, values, labels)
        return self

    def predict(
        self,
        values: Data,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]:
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
        result : Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just prediction, otherwise a tuple 
            will be returned with first element being prediction and second one being metrics.
        """
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_result(result_example_set)
        if return_metrics:
            metrics_values: dict = BaseClassifier._calculate_prediction_metrics(
                self, result_example_set)
            return (mapped_result_example_set, metrics_values)
        else:
            return mapped_result_example_set

    def predict_proba(
        self,
        values: Data,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]:
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
        result : Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just probabilities matrix, otherwise
            a tuple will be returned with first element being prediction and second one being 
            metrics.
        """
        result_example_set = BaseOperator.predict(self, values)
        mapped_result_example_set = self._map_confidence(result_example_set)
        if return_metrics:
            metrics_values: dict = BaseClassifier._calculate_prediction_metrics(
                self, result_example_set)
            return (mapped_result_example_set, metrics_values)
        else:
            return mapped_result_example_set

    def score(self, values: Data, labels: Data) -> float:
        """Return the accuracy on the given test data and labels.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            true labels

        Returns
        -------
        score : float
            Accuracy of self.predict(values) wrt. labels.
        """
        predicted_labels = self.predict(values)
        return metrics.accuracy_score(labels, predicted_labels)

    def __getstate__(self) -> dict:
        return {**BaseOperator.__getstate__(self), **{
            'label_unique_values': self.label_unique_values,
            '_remap_to_numeric': self._remap_to_numeric
        }}

    def __setstate__(self, state: dict):
        BaseOperator.__setstate__(self, state)
        self._init_classification_rule_performance()
        self.label_unique_values = state['label_unique_values']
        self._remap_to_numeric = state['_remap_to_numeric']


class ExpertRuleClassifier(ExpertKnowledgeOperator, RuleClassifier):
    """Classification model using expert knowledge."""

    __params_class__ = ClassificationParams

    def __init__(
        self,
        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE['induction_measure'],
        pruning_measure: Union[Measures,
                               str] = DEFAULT_PARAMS_VALUE['pruning_measure'],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE['voting_measure'],
        max_growing: float = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],
        control_apriori_precision: bool = DEFAULT_PARAMS_VALUE['control_apriori_precision'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
        approximate_induction: bool = DEFAULT_PARAMS_VALUE['approximate_induction'],
        approximate_bins_count: int = DEFAULT_PARAMS_VALUE['approximate_bins_count'],

        extend_using_preferred: bool = DEFAULT_PARAMS_VALUE['extend_using_preferred'],
        extend_using_automatic: bool = DEFAULT_PARAMS_VALUE['extend_using_automatic'],
        induce_using_preferred: bool = DEFAULT_PARAMS_VALUE['induce_using_preferred'],
        induce_using_automatic: bool = DEFAULT_PARAMS_VALUE['induce_using_automatic'],
        consider_other_classes: bool = DEFAULT_PARAMS_VALUE['consider_other_classes'],
        preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
            'preferred_conditions_per_rule'],
        preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE[
            'preferred_attributes_per_rule'],
        min_rule_covered: Optional[int] = None
    ):
        """
        Parameters
        ----------
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be
            covered by a new rule (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during pruning. Could be user defined (string), for example  
            :code:`2 * p / n`; default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during voting; default measure is correlation
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to
            the rule in the growing phase (use this parameter for large datasets if execution time
            is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value
            of given attribute is always considered as not fulfilling the condition build upon 
            that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default: 
            False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        control_apriori_precision : bool = True
            When inducing classification rules, verify if candidate precision is higher than 
            apriori precision of the investigated class.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.
        approximate_induction: bool = False
            Use an approximate induction heuristic which does not check all possible splits;
            note: this is an experimental feature and currently works only for classification
            data sets, results may change in future;
        approximate_bins_count: int = 100
            maximum number of bins for an attribute evaluated in the approximate induction.

        extend_using_preferred : bool = False
            boolean indicating whether initial rules should be extended with a use of preferred
            conditions and attributes; default is False
        extend_using_automatic : bool = False
            boolean indicating whether initial rules should be extended with a use of automatic 
            conditions and attributes; default is False
        induce_using_preferred : bool = False
            boolean indicating whether new rules should be induced with a use of preferred 
            conditions and attributes; default is False
        induce_using_automatic : bool = False
            boolean indicating whether new rules should be induced with a use of automatic 
            conditions and attributes; default is False
        consider_other_classes : bool = False
            boolean indicating whether automatic induction should be performed for classes for
              which no user's knowledge has been defined (classification only); default is False.
        preferred_conditions_per_rule : int = None
            maximum number of preferred conditions per rule; default: unlimited,
        preferred_attributes_per_rule : int = None
            maximum number of preferred attributes per rule; default: unlimited.
        min_rule_covered : int = None
            alias to `minsupp_new`. Parameter is deprecated and will be removed in the next major
            version, use `minsupp_new`

            .. deprecated:: 1.7.0
                Use parameter `minsupp_new` instead.
        """
        self._remap_to_numeric = False
        RuleClassifier.__init__(
            self,
            minsupp_new=minsupp_new,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            control_apriori_precision=control_apriori_precision,
            max_rule_count=max_rule_count,
            approximate_induction=approximate_induction,
            approximate_bins_count=approximate_bins_count,
        )
        ExpertKnowledgeOperator.__init__(
            self,
            minsupp_new=minsupp_new,
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            extend_using_preferred=extend_using_preferred,
            extend_using_automatic=extend_using_automatic,
            induce_using_preferred=induce_using_preferred,
            induce_using_automatic=induce_using_automatic,
            consider_other_classes=consider_other_classes,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule,
            control_apriori_precision=control_apriori_precision,
            max_rule_count=max_rule_count,
            approximate_induction=approximate_induction,
            approximate_bins_count=approximate_bins_count,
        )

    def fit(  # pylint: disable=arguments-differ
        self,
        values: Data,
        labels: Data,

        expert_rules: list[Union[str, tuple[str, str]]] = None,
        expert_preferred_conditions: list[Union[str, tuple[str, str]]] = None,
        expert_forbidden_conditions: list[Union[str, tuple[str, str]]] = None
    ) -> ExpertRuleClassifier:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            labels

        expert_rules : List[Union[str, Tuple[str, str]]]
            set of initial rules, either passed as a list of strings representing rules or as list
            of tuples where first element is name of the rule and second one is rule string.
        expert_preferred_conditions : List[Union[str, Tuple[str, str]]]
            multiset of preferred conditions (used also for specifying preferred attributes by
            using special value Any). Either passed as a list of strings representing rules or as
            list of tuples where first element is name of the rule and second one is rule string.
        expert_forbidden_conditions : List[Union[str, Tuple[str, str]]]
            set of forbidden conditions (used also for specifying forbidden attributes by using 
            special valye Any). Either passed as a list of strings representing rules or as list
            of tuples where first element is name of the rule and second one is rule string.
        Returns
        -------
        self : ExpertRuleClassifier
        """
        if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
            if isinstance(labels.iloc[0], Number):
                self._remap_to_numeric = True
                labels = labels.astype(str)
        else:
            if isinstance(labels[0], Number):
                self._remap_to_numeric = True
                labels = list(map(str, labels))
        self._get_unique_label_values(labels)
        self._prepare_labels(labels)
        return ExpertKnowledgeOperator.fit(
            self,
            values,
            labels,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def predict(
        self,
        values: Data,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]:
        return RuleClassifier.predict(self, values, return_metrics)

    def __getstate__(self) -> dict:
        return {**BaseOperator.__getstate__(self), **{
            '_remap_to_numeric': self._remap_to_numeric
        }}

    def __setstate__(self, state: dict):
        BaseOperator.__setstate__(self, state)
        self._remap_to_numeric = state['_remap_to_numeric']


class ContrastSetRuleClassifier(BaseOperator, BaseClassifier):
    """Contrast set classification model."""

    __params_class__ = ContrastSetModelParams

    def __init__(
        self,
        minsupp_all: Iterable[float] = DEFAULT_PARAMS_VALUE['minsupp_all'],
        max_neg2pos: float = DEFAULT_PARAMS_VALUE['max_neg2pos'],
        max_passes_count: int = DEFAULT_PARAMS_VALUE['max_passes_count'],
        penalty_strength: float = DEFAULT_PARAMS_VALUE['penalty_strength'],
        penalty_saturation: float = DEFAULT_PARAMS_VALUE['penalty_saturation'],

        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE['induction_measure'],
        pruning_measure: Union[Measures,
                               str] = DEFAULT_PARAMS_VALUE['pruning_measure'],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE['voting_measure'],
        max_growing: float = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],
        control_apriori_precision: bool = DEFAULT_PARAMS_VALUE['control_apriori_precision'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
        approximate_induction: bool = DEFAULT_PARAMS_VALUE['approximate_induction'],
        approximate_bins_count: int = DEFAULT_PARAMS_VALUE['approximate_bins_count'],
    ):
        """
        Parameters
        ----------
        minsupp_all: Iterable[float]
            a minimum positive support of a contrast set (p/P). When multiple values are specified,
            a metainduction is performed; Default and recommended sequence is: 0.8, 0.5, 0.2, 0.1
        max_neg2pos: float
            a maximum ratio of negative to positive supports (nP/pN); Default is 0.5
        max_passes_count: int
            a maximum number of sequential covering passes for a single minsupp-all; Default is 5
        penalty_strength: float
            (s) - penalty strength; Default is 0.5
        penalty_saturation: float
            the value of p_new / P at which penalty reward saturates; Default is 0.2.
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be 
            covered by a new rule (positive examples for classification problems); default: 5
        induction_measure : :class:`rulekit.params.Measures` = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during induction; default measure is correlation
        pruning_measure : Union[:class:`rulekit.params.Measures`, str] = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during pruning. Could be user defined (string), for example  
            :code:`2 * p / n`; default measure is correlation
        voting_measure : :class:`rulekit.params.Measures` = \
            :class:`rulekit.params.Measures.Correlation`
            measure used during voting; default measure is correlation
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to
            the rule in the growing phase (use this parameter for large datasets if execution time
            is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value
            of given attribute is always considered as not fulfilling the condition build upon 
            that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples 
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; 
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        control_apriori_precision : bool = True
            When inducing classification rules, verify if candidate precision is higher than 
            apriori precision of the investigated class.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.
        approximate_induction: bool = False
            Use an approximate induction heuristic which does not check all possible splits;
            note: this is an experimental feature and currently works only for classification
            data sets, results may change in future;
        approximate_bins_count: int = 100
            maximum number of bins for an attribute evaluated in the approximate induction.
        """
        BaseOperator.__init__(
            self,
            minsupp_all=minsupp_all,
            max_neg2pos=max_neg2pos,
            max_passes_count=max_passes_count,
            penalty_strength=penalty_strength,
            penalty_saturation=penalty_saturation,
            minsupp_new=minsupp_new,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            control_apriori_precision=control_apriori_precision,
            max_rule_count=max_rule_count,
            approximate_induction=approximate_induction,
            approximate_bins_count=approximate_bins_count
        )
        BaseClassifier.__init__(self)
        self.contrast_attribute: str = None
        self._remap_to_numeric = False
        self.label_unique_values = []

    def _map_result(self, predicted_example_set) -> np.ndarray:
        prediction: np.ndarray
        if self._remap_to_numeric:
            prediction = PredictionResultMapper.map_to_numerical(
                predicted_example_set)
        else:
            prediction = PredictionResultMapper.map_to_nominal(
                predicted_example_set)
        return prediction

    def _get_unique_label_values(self, labels: Data):
        tmp = {}
        for label_value in labels:
            tmp[label_value] = None
        self.label_unique_values = list(tmp.keys())
        if len(self.label_unique_values) > 0 and isinstance(self.label_unique_values[0], bytes):
            self.label_unique_values = [item.decode(
                'utf-8') for item in self.label_unique_values]

    def fit(self, values: Data, labels: Data, contrast_attribute: str) -> ContrastSetRuleClassifier:  # pylint: disable=arguments-differ
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            labels
        contrast_attribute: str 
            group attribute
        Returns
        -------
        self : ContrastSetRuleClassifier
        """
        RuleClassifier._get_unique_label_values(  # pylint: disable=protected-access
            self, labels
        )
        RuleClassifier._prepare_labels(  # pylint: disable=protected-access,protected-access
            self, labels
        )
        BaseOperator.fit(
            self, values, labels,
            contrast_attribute=contrast_attribute
        )
        self.contrast_attribute = contrast_attribute
        return self

    def predict(
        self,
        values: Data,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]:
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
        result : Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just prediction, otherwise a tuple
            will be returned with first element being prediction and second one being metrics.
        """
        return RuleClassifier.predict(self, values, return_metrics)

    def predict_proba(
        self,
        values: Data,
        return_metrics: bool = False
    ) -> Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]:
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
        result : Union[np.ndarray, tuple[np.ndarray, dict[str, float]]]
            If *return_metrics* flag wasn't set it will return just probabilities matrix, otherwise
            a tuple will be returned with first element being prediction and second one being 
            metrics.
        """
        return RuleClassifier.predict_proba(self, values, return_metrics)

    def score(self, values: Data, labels: Data) -> float:
        """Return the accuracy on the given test data and labels.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            true labels

        Returns
        -------
        score : float
            Accuracy of self.predict(values) wrt. labels.
        """
        return RuleClassifier.score(self, values, labels)

    def __getstate__(self) -> dict:
        return {**BaseOperator.__getstate__(self), **{
            'label_unique_values': self.label_unique_values,
            '_remap_to_numeric': self._remap_to_numeric,
            'contrast_attribute': self.contrast_attribute,
        }}

    def __setstate__(self, state: dict):
        BaseOperator.__setstate__(self, state)
        self._init_classification_rule_performance()
        self.label_unique_values = state['label_unique_values']
        self._remap_to_numeric = state['_remap_to_numeric']
        self.contrast_attribute = state['contrast_attribute']
