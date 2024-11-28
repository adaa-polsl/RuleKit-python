"""Module containing classes for regression analysis and prediction.
"""
from __future__ import annotations

from numbers import Number
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sklearn import metrics

from rulekit._helpers import PredictionResultMapper
from rulekit._operator import BaseOperator
from rulekit._operator import Data
from rulekit._operator import ExpertKnowledgeOperator
from rulekit._problem_types import ProblemType
from rulekit.params import ContrastSetModelParams
from rulekit.params import DEFAULT_PARAMS_VALUE
from rulekit.params import ExpertModelParams
from rulekit.params import Measures
from rulekit.params import ModelsParams
from rulekit.rules import RegressionRule
from rulekit.rules import RuleSet


class _RegressionModelParams(ModelsParams):
    mean_based_regression: bool = DEFAULT_PARAMS_VALUE["mean_based_regression"]


class _RegressionExpertModelParams(_RegressionModelParams, ExpertModelParams):
    pass


class RuleRegressor(BaseOperator):
    """Regression model."""

    __params_class__ = _RegressionModelParams

    def __init__(  # pylint: disable=too-many-arguments
        self,
        minsupp_new: float = DEFAULT_PARAMS_VALUE["minsupp_new"],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE["induction_measure"],
        pruning_measure: Union[Measures, str] = DEFAULT_PARAMS_VALUE["pruning_measure"],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE["voting_measure"],
        max_growing: float = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
        mean_based_regression: bool = DEFAULT_PARAMS_VALUE["mean_based_regression"],
        max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"],
    ):
        """
        Parameters
        ----------
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples to be covered by a new rule (positive examples for classification
            problems); default: 5,
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
            non-negative integer representing maximum number of conditions which can be
            added to the rule in the growing phase (use this parameter for large
            datasets if execution time is prohibitive); 0 indicates no limit; default: 0
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a
            missing value of given attribute is always considered as not fulfilling the
            condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of
            examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase;
             default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal
            attributes are supported.
        mean_based_regression : bool = True
            Enable fast induction of mean-based regression rules instead of default
            median-based.
        max_rule_count : int = 0
            Maximum number of rules to be generated; 0 indicates no limit. Default: 0
        """
        super().__init__(
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
            mean_based_regression=mean_based_regression,
            max_rule_count=max_rule_count,
        )
        self.model: RuleSet[RegressionRule] = None

    def _validate_labels(self, labels: Data):
        if isinstance(labels, (pd.DataFrame, pd.Series)):
            first_label = labels.iloc[0]
        else:
            first_label = labels[0]
        if not isinstance(first_label, Number):
            raise ValueError(
                f"{self.__class__.__name__} requires lables values to be numeric"
            )

    def fit(  # pylint: disable=arguments-differ
        self, values: Data, labels: Data
    ) -> RuleRegressor:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            target values
        Returns
        -------
        self : RuleRegressor
        """
        self._validate_labels(labels)
        super().fit(values, labels)
        return self

    def predict(self, values: Data) -> np.ndarray:
        """Perform prediction and returns predicted values.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        Returns
        -------
        result : np.ndarray
            predicted values
        """
        return self._map_result(super().predict(values))

    def score(self, values: Data, labels: Data) -> float:
        """Return the coefficient of determination R2 of the prediction

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            true target values

        Returns
        -------
        score : float
            R2 of self.predict(values) wrt. labels.
        """
        predicted_labels = self.predict(values)
        return metrics.r2_score(labels, predicted_labels)

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map_to_numerical(
            predicted_example_set, remap=False
        )

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.REGRESSION


class ExpertRuleRegressor(ExpertKnowledgeOperator, RuleRegressor):
    """Expert Regression model."""

    __params_class__ = _RegressionExpertModelParams

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        minsupp_new: float = DEFAULT_PARAMS_VALUE["minsupp_new"],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE["induction_measure"],
        pruning_measure: Union[Measures, str] = DEFAULT_PARAMS_VALUE["pruning_measure"],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE["voting_measure"],
        max_growing: float = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
        mean_based_regression: bool = DEFAULT_PARAMS_VALUE["mean_based_regression"],
        max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"],
        extend_using_preferred: bool = DEFAULT_PARAMS_VALUE["extend_using_preferred"],
        extend_using_automatic: bool = DEFAULT_PARAMS_VALUE["extend_using_automatic"],
        induce_using_preferred: bool = DEFAULT_PARAMS_VALUE["induce_using_preferred"],
        induce_using_automatic: bool = DEFAULT_PARAMS_VALUE["induce_using_automatic"],
        preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
            "preferred_conditions_per_rule"
        ],
        preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE[
            "preferred_attributes_per_rule"
        ],
    ):
        """
        Parameters
        ----------
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples  to be covered by a new rule (positive examples for classification
            problems); default: 5,
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
            non-negative integer representing maximum number of conditions which can be
            added to the rule in the growing phase (use this parameter for large
            datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a
            missing value of given attribute is always considered as not fulfilling the
            condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of
            examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase;
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal
            attributes are supported.
        mean_based_regression : bool = True
            Enable fast induction of mean-based regression rules instead of default
            median-based.
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it
            applies to a single class); 0 indicates no limit.

        extend_using_preferred : bool = False
            boolean indicating whether initial rules should be extended with a use of
            preferred conditions and attributes; default is False
        extend_using_automatic : bool = False
            boolean indicating whether initial rules should be extended with a use of
            automatic conditions and attributes; default is False
        induce_using_preferred : bool = False
            boolean indicating whether new rules should be induced with a use of
            preferred conditions and attributes; default is False
        induce_using_automatic : bool = False
            boolean indicating whether new rules should be induced with a use of
            automatic conditions and attributes; default is False
        preferred_conditions_per_rule : int = None
            maximum number of preferred conditions per rule; default: unlimited,
        preferred_attributes_per_rule : int = None
            maximum number of preferred attributes per rule; default: unlimited.
        """
        RuleRegressor.__init__(
            self,
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
            mean_based_regression=mean_based_regression,
            max_rule_count=max_rule_count,
        )
        ExpertKnowledgeOperator.__init__(
            self,
            minsupp_new=minsupp_new,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
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
            mean_based_regression=mean_based_regression,
            max_rule_count=max_rule_count,
        )
        self.model: RuleSet[RegressionRule] = None

    def fit(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        values: Data,
        labels: Data,
        expert_rules: list[Union[str, tuple[str, str]]] = None,
        expert_preferred_conditions: list[Union[str, tuple[str, str]]] = None,
        expert_forbidden_conditions: list[Union[str, tuple[str, str]]] = None,
    ) -> ExpertRuleRegressor:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            target values

        expert_rules : List[Union[str, Tuple[str, str]]]
            set of initial rules, either passed as a list of strings representing rules
            or as list of tuples where first element is name of the rule and second one
            is rule string.
        expert_preferred_conditions : List[Union[str, Tuple[str, str]]]
            multiset of preferred conditions (used also for specifying preferred
            attributes by using special value Any). Either passed as a list of strings
            representing rules or as list of tuples where first element is name of the
            rule and second one is rule string.
        expert_forbidden_conditions : List[Union[str, Tuple[str, str]]]
            set of forbidden conditions (used also for specifying forbidden attributes
            by using special valye Any). Either passed as a list of strings representing
            rules or as list of tuples where first element is name of the rule and
            second one is rule string.
        Returns
        -------
        self : ExpertRuleRegressor
        """
        self._validate_labels(labels)
        return ExpertKnowledgeOperator.fit(
            self,
            values,
            labels,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions,
        )

    def predict(self, values: Data) -> np.ndarray:
        return self._map_result(ExpertKnowledgeOperator.predict(self, values))

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.REGRESSION


class ContrastSetRuleRegressor(BaseOperator):
    """Contrast set regression model."""

    __params_class__ = ContrastSetModelParams

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        minsupp_all: Tuple[float, float, float, float] = DEFAULT_PARAMS_VALUE[
            "minsupp_all"
        ],
        max_neg2pos: float = DEFAULT_PARAMS_VALUE["max_neg2pos"],
        max_passes_count: int = DEFAULT_PARAMS_VALUE["max_passes_count"],
        penalty_strength: float = DEFAULT_PARAMS_VALUE["penalty_strength"],
        penalty_saturation: float = DEFAULT_PARAMS_VALUE["penalty_saturation"],
        minsupp_new: float = DEFAULT_PARAMS_VALUE["minsupp_new"],
        induction_measure: Measures = DEFAULT_PARAMS_VALUE["induction_measure"],
        pruning_measure: Union[Measures, str] = DEFAULT_PARAMS_VALUE["pruning_measure"],
        voting_measure: Measures = DEFAULT_PARAMS_VALUE["voting_measure"],
        max_growing: float = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
        mean_based_regression: bool = DEFAULT_PARAMS_VALUE["mean_based_regression"],
        max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"],
    ):
        """
        Parameters
        ----------
        minsupp_all: Tuple[float, float, float, float]
            a minimum positive support of a contrast set (p/P). When multiple values are
            specified, a metainduction is performed; Default and recommended sequence
            is: 0.8, 0.5, 0.2, 0.1
        max_neg2pos: float
            a maximum ratio of negative to positive supports (nP/pN); Default is 0.5
        max_passes_count: int
            a maximum number of sequential covering passes for a single minsupp-all;
            Default is 5
        penalty_strength: float
            (s) - penalty strength; Default is 0.5
        penalty_saturation: float
            the value of p_new / P at which penalty reward saturates; Default is 0.2.
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples to be covered by a new rule (positive examples for classification
            problems); default: 5,
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
            non-negative integer representing maximum number of conditions which can be
            added to the rule in the growing phase (use this parameter for large
            datasets if execution time is prohibitive); 0 indicates no limit; default: 0
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a
            missing value of given attribute is always considered as not fulfilling the
            condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of
            examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase;
            default: False.
        complementary_conditions : bool = False
            If enabled, complementary conditions in the form a = !{value} for nominal
            attributes are supported.
        mean_based_regression : bool = True
            Enable fast induction of mean-based regression rules instead of default
            median-based.
        max_rule_count : int = 0
            Maximum number of rules to be generated; 0 indicates no limit.
        """
        super().__init__(
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
            mean_based_regression=mean_based_regression,
            max_rule_count=max_rule_count,
        )
        self.contrast_attribute: str = None
        self.model: RuleSet[RegressionRule] = None

    def fit(
        self, values: Data, labels: Data, contrast_attribute: str
    ) -> ContrastSetRuleRegressor:  # pylint: disable=arguments-differ
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            target values
        contrast_attribute: str
            group attribute
        Returns
        -------
        self : ContrastSetRuleRegressor
        """
        RuleRegressor._validate_labels(self, labels)  # pylint: disable=protected-access
        super().fit(values, labels, contrast_attribute=contrast_attribute)
        self.contrast_attribute = contrast_attribute
        return self

    def predict(self, values: Data) -> np.ndarray:
        """Perform prediction and returns predicted values.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes

        Returns
        -------
        result : np.ndarray
            predicted values
        """
        return RuleRegressor.predict(self, values)

    def score(self, values: Data, labels: Data) -> float:
        """Return the coefficient of determination R2 of the prediction

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            true target values

        Returns
        -------
        score : float
            R2 of self.predict(values) wrt. labels.
        """
        return RuleRegressor.score(self, values, labels)

    def __getstate__(self) -> dict:
        return {
            **BaseOperator.__getstate__(self),
            **{
                "contrast_attribute": self.contrast_attribute,
            },
        }

    def __setstate__(self, state: dict):
        BaseOperator.__setstate__(self, state)
        self.contrast_attribute = state["contrast_attribute"]

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.CONTRAST_REGRESSION
