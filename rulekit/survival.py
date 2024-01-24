"""Module contaiing classes for survival analysis and prediction.
"""
from __future__ import annotations
from typing import Optional, Union, Iterable
import numpy as np
import pandas as pd
from jpype import JClass
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from ._helpers import (
    PredictionResultMapper,
    get_rule_generator,
    create_example_set
)
from ._operator import BaseOperator, ExpertKnowledgeOperator, Data
from .params import ContrastSetModelParams, DEFAULT_PARAMS_VALUE
from .rules import RuleSet


class SurvivalModelsParams(BaseModel):
    """Model for validating survival models hyperparameters
    """
    survival_time_attr: Optional[str]
    minsupp_new: Optional[int] = DEFAULT_PARAMS_VALUE['minsupp_new']
    max_growing: Optional[float] = DEFAULT_PARAMS_VALUE['max_growing']
    enable_pruning: Optional[bool] = DEFAULT_PARAMS_VALUE['enable_pruning']
    ignore_missing: Optional[bool] = DEFAULT_PARAMS_VALUE['ignore_missing']
    max_uncovered_fraction: Optional[float] = DEFAULT_PARAMS_VALUE['max_uncovered_fraction']
    select_best_candidate: Optional[bool] = DEFAULT_PARAMS_VALUE['select_best_candidate']
    min_rule_covered: Optional[int] = None
    complementary_conditions: Optional[bool] = DEFAULT_PARAMS_VALUE['complementary_conditions']

    extend_using_preferred: Optional[bool] = None
    extend_using_automatic: Optional[bool] = None
    induce_using_preferred: Optional[bool] = None
    induce_using_automatic: Optional[bool] = None
    consider_other_classes: Optional[bool] = None
    preferred_conditions_per_rule: Optional[int] = None
    preferred_attributes_per_rule: Optional[int] = None


class SurvivalRules(BaseOperator):
    """Survival model."""

    __params_class__ = SurvivalModelsParams

    def __init__(  # pylint: disable=super-init-not-called
        self,
        survival_time_attr: str = None,
        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
        min_rule_covered: Optional[int] = None
    ):
        """
        Parameters
        ----------
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model is padnas
            dataframe).
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be
            covered by a new rule (positive examples for classification problems); default: 5
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to
            the rule in the growing phase  (use this parameter for large datasets if execution time
            is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value 
            of given attribute is always considered as not fulfilling the condition build upon that
            attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples 
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; 
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.
        min_rule_covered : int = None
            alias to `minsupp_new`. Parameter is deprecated and will be removed in the next major
            version, use `minsupp_new`

            .. deprecated:: 1.7.0
                Use parameter `minsupp_new` instead.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.set_params(
            survival_time_attr=survival_time_attr,
            minsupp_new=minsupp_new,
            min_rule_covered=min_rule_covered,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            max_rule_count=max_rule_count
        )
        self.model: RuleSet = None

    def set_params(
        self,
        **kwargs
    ) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self.survival_time_attr = kwargs.get('survival_time_attr')
        return BaseOperator.set_params(self, **kwargs)

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
            raise ValueError(
                'Data values must be instance of either pandas DataFrame, numpy array or list'
            )
        return ''

    def _prepare_survival_attribute(self, survival_time: Data, values: Data) -> str:
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError(
                'No "survival_time" attribute name was specified. ' +
                'Specify it using method set_params'
            )
        if survival_time is not None:
            return SurvivalRules._append_survival_time_columns(
                values, survival_time)
        else:
            return self.survival_time_attr

    def fit(self, values: Data, labels: Data, survival_time: Data = None) -> SurvivalRules:  # pylint: disable=arguments-differ
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter 
            was specified.

        Returns
        -------
        self : SurvivalRules
        """
        survival_time_attribute = self._prepare_survival_attribute(
            survival_time, values)
        super().fit(values, labels, survival_time_attribute)
        return self

    def predict(self, values: Data) -> np.ndarray:
        """Perform prediction and return estimated survival function for each example.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        Returns
        -------
        result : np.ndarray
            Each row represent single example from dataset and contains estimated survival function
            for that example. Estimated survival function is returned as a dictionary containing 
            times and corresponding probabilities.
        """
        return PredictionResultMapper.map_survival(super().predict(values))

    def score(self, values: Data, labels: Data, survival_time: Data = None) -> float:
        """Return the Integrated Brier Score on the given dataset and labels 
        (event status indicator).

        Integrated Brier Score (IBS) - the Brier score (BS) represents the squared difference 
        between true event status at time T and predicted event status at that time; 
        the Integrated Brier score summarizes the prediction error over all observations and over
        all times in a test set.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was
            specified

        Returns
        -------
        score : float
            Integrated Brier Score of self.predict(values) wrt. labels.
        """

        survival_time_attribute = self._prepare_survival_attribute(
            survival_time, values)
        example_set = create_example_set(
            values, labels,  survival_time_attribute=survival_time_attribute)

        predicted_example_set = self.model._java_object.apply(  # pylint: disable=protected-access
            example_set
        )

        IntegratedBrierScore = JClass(  # pylint: disable=invalid-name
            'adaa.analytics.rules.logic.quality.IntegratedBrierScore'
        )
        integrated_brier_score = IntegratedBrierScore()
        integrated_brier_score.startCounting(predicted_example_set, True)
        return integrated_brier_score.getMikroAverage()


class ExpertSurvivalRules(ExpertKnowledgeOperator, SurvivalRules):
    """Expert Survival model."""

    __params_class__ = SurvivalModelsParams

    def __init__(  # pylint: disable=super-init-not-called
        self,
        survival_time_attr: str = None,
        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],

        extend_using_preferred: bool = DEFAULT_PARAMS_VALUE['extend_using_preferred'],
        extend_using_automatic: bool = DEFAULT_PARAMS_VALUE['extend_using_automatic'],
        induce_using_preferred: bool = DEFAULT_PARAMS_VALUE['induce_using_preferred'],
        induce_using_automatic: bool = DEFAULT_PARAMS_VALUE['induce_using_automatic'],
        preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
            'preferred_conditions_per_rule'],
        preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE[
            'preferred_attributes_per_rule'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
        min_rule_covered: Optional[int] = None
    ):
        """
        Parameters
        ----------
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be
            covered by a new rule (positive examples for classification problems); default: 5
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model is pandas
            dataframe).
        min_rule_covered : int = 5
            positive integer representing minimum number of previously uncovered examples to be 
            covered by a new rule (positive examples for classification problems); default: 5
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to
            the rule in the growing phase (use this parameter for large datasets if execution time
            is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value
            of given attribute is always considered as not fulfilling the condition build upon that
            attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples 
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; 
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.

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
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.set_params(
            survival_time_attr=survival_time_attr,
            minsupp_new=minsupp_new,
            min_rule_covered=min_rule_covered,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            extend_using_preferred=extend_using_preferred,
            extend_using_automatic=extend_using_automatic,
            induce_using_preferred=induce_using_preferred,
            induce_using_automatic=induce_using_automatic,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule,
            complementary_conditions=complementary_conditions,
            max_rule_count=max_rule_count
        )
        self.model: RuleSet = None

    def set_params(  # pylint: disable=arguments-differ
        self,
        **kwargs
    ) -> object:
        self.survival_time_attr = kwargs['survival_time_attr']
        return ExpertKnowledgeOperator.set_params(self, **kwargs)

    def fit(  # pylint: disable=arguments-differ
        self,
        values: Data,
        labels: Data,
        survival_time: Data = None,

        expert_rules: list[Union[str, tuple[str, str]]] = None,
        expert_preferred_conditions: list[Union[str, tuple[str, str]]] = None,
        expert_forbidden_conditions: list[Union[str, tuple[str, str]]] = None
    ) -> ExpertSurvivalRules:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : Data
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was
            specified.

        expert_rules : List[Union[str, Tuple[str, str]]]
            set of initial rules, either passed as a list of strings representing rules or as list
            of tuples where first
            element is name of the rule and second one is rule string.
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
        self : ExpertSurvivalRules
        """
        survival_time_attribute = SurvivalRules._prepare_survival_attribute(
            self, survival_time, values)
        return ExpertKnowledgeOperator.fit(
            self,
            values=values,
            labels=labels,
            survival_time_attribute=survival_time_attribute,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions
        )

    def predict(self, values: Data) -> np.ndarray:
        return PredictionResultMapper.map_survival(ExpertKnowledgeOperator.predict(self, values))


class SurvivalContrastSetModelParams(ContrastSetModelParams, SurvivalModelsParams):
    """Model for validating survival contrast sets models hyperparameters
     """


class ContrastSetSurvivalRules(BaseOperator):
    """Contrast set survival model."""

    __params_class__ = SurvivalContrastSetModelParams

    def __init__(  # pylint: disable=super-init-not-called
        self,
        minsupp_all: Iterable[float] = DEFAULT_PARAMS_VALUE['minsupp_all'],
        max_neg2pos: float = DEFAULT_PARAMS_VALUE['max_neg2pos'],
        max_passes_count: int = DEFAULT_PARAMS_VALUE['max_passes_count'],
        penalty_strength: float = DEFAULT_PARAMS_VALUE['penalty_strength'],
        penalty_saturation: float = DEFAULT_PARAMS_VALUE['penalty_saturation'],

        survival_time_attr: str = None,
        minsupp_new: int = DEFAULT_PARAMS_VALUE['minsupp_new'],
        max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE['complementary_conditions'],
        max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count'],
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
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model is pandas
            dataframe).
        minsupp_new : int = 5
            positive integer representing minimum number of previously uncovered examples to be 
            covered by a new rule  (positive examples for classification problems); default: 5
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to 
            the rule in the growing phase (use this parameter for large datasets if execution time
            is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value 
            of given attribute is always considered as not fulfilling the condition build upon that
            attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples 
            that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase;
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal attributes
            are supported.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it applies 
            to a single class); 0 indicates no limit.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.contrast_attribute: str = None
        self.set_params(
            minsupp_all=minsupp_all,
            max_neg2pos=max_neg2pos,
            max_passes_count=max_passes_count,
            penalty_strength=penalty_strength,
            penalty_saturation=penalty_saturation,
            survival_time_attr=survival_time_attr,
            minsupp_new=minsupp_new,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            max_rule_count=max_rule_count
        )
        self.model: RuleSet = None

    def set_params(self, **kwargs) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        # params validation
        self.survival_time_attr = kwargs['survival_time_attr']
        return BaseOperator.set_params(self, **kwargs)

    def fit(  # pylint: disable=arguments-renamed
        self,
        values: Data,
        labels: Data,
        contrast_attribute: str,
        survival_time: Data = None
    ) -> ContrastSetSurvivalRules:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        contrast_attribute: str 
            group attribute
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter 
            was specified.

        Returns
        -------
        self : ContrastSetSurvivalRules
        """
        survival_time_attribute = SurvivalRules._prepare_survival_attribute(  # pylint: disable=protected-access
            self, survival_time, values)
        super().fit(
            values, labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute)
        self.contrast_attribute = contrast_attribute
        return self

    def predict(self, values: Data) -> np.ndarray:
        """Perform prediction and return estimated survival function for each example.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        Returns
        -------
        result : np.ndarray
            Each row represent single example from dataset and contains estimated survival function
            for that example. Estimated survival function is returned as a dictionary containing 
            times and corresponding probabilities.
        """
        return PredictionResultMapper.map_survival(super().predict(values))

    def score(self, values: Data, labels: Data, survival_time: Data = None) -> float:
        """Return the Integrated Brier Score on the given dataset and 
        labels(event status indicator).

        Integrated Brier Score (IBS) - the Brier score (BS) represents the squared difference
        between true event status at time T and predicted event status at that time; 
        the Integrated Brier score summarizes the prediction error over all observations and 
        over all times in a test set.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was
            specified

        Returns
        -------
        score : float
            Integrated Brier Score of self.predict(values) wrt. labels.
        """
        return SurvivalRules.score(self, values, labels, survival_time=survival_time)
