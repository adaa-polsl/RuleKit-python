"""Module containing classes for survival analysis and prediction.
"""
from __future__ import annotations

from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from jpype import JClass
from pydantic import BaseModel  # pylint: disable=no-name-in-module

from rulekit._helpers import ExampleSetFactory
from rulekit._helpers import PredictionResultMapper
from rulekit._operator import BaseOperator
from rulekit._operator import Data
from rulekit._operator import ExpertKnowledgeOperator
from rulekit._problem_types import ProblemType
from rulekit.kaplan_meier import KaplanMeierEstimator
from rulekit.params import ContrastSetModelParams
from rulekit.params import DEFAULT_PARAMS_VALUE
from rulekit.params import ExpertModelParams
from rulekit.rules import RuleSet
from rulekit.rules import SurvivalRule

_DEFAULT_SURVIVAL_TIME_ATTR: str = "survival_time"


class _SurvivalModelsParams(BaseModel):
    survival_time_attr: Optional[str]
    minsupp_new: Optional[float] = DEFAULT_PARAMS_VALUE["minsupp_new"]
    max_growing: Optional[float] = DEFAULT_PARAMS_VALUE["max_growing"]
    enable_pruning: Optional[bool] = DEFAULT_PARAMS_VALUE["enable_pruning"]
    ignore_missing: Optional[bool] = DEFAULT_PARAMS_VALUE["ignore_missing"]
    max_uncovered_fraction: Optional[float] = DEFAULT_PARAMS_VALUE[
        "max_uncovered_fraction"
    ]
    select_best_candidate: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "select_best_candidate"
    ]
    complementary_conditions: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "complementary_conditions"
    ]


class _SurvivalExpertModelParams(_SurvivalModelsParams, ExpertModelParams):
    pass


class _BaseSurvivalRulesModel:

    model: RuleSet[SurvivalRule]

    def get_train_set_kaplan_meier(self) -> KaplanMeierEstimator:
        """Returns train set KaplanMeier estimator

        Returns:
            KaplanMeierEstimator: estimator
        """
        return KaplanMeierEstimator(
            self.model._java_object.getTrainingEstimator()  # pylint: disable=protected-access
        )


class SurvivalRules(BaseOperator, _BaseSurvivalRulesModel):
    """Survival model."""

    __params_class__ = _SurvivalModelsParams

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        survival_time_attr: str = None,
        minsupp_new: int = DEFAULT_PARAMS_VALUE["minsupp_new"],
        max_growing: int = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
        max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"],
    ):
        """
        Parameters
        ----------
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model
            is padnas dataframe).
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples to be covered by a new rule (positive examples for classification
            problems); default: 5,
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be
            added to the rule in the growing phase  (use this parameter for large
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
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it
            applies to a single class); 0 indicates no limit.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self._initialize_rulekit()
        self.set_params(
            survival_time_attr=survival_time_attr,
            minsupp_new=minsupp_new,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate,
            complementary_conditions=complementary_conditions,
            max_rule_count=max_rule_count,
        )
        self.model: RuleSet[SurvivalRule] = None

    def set_params(self, **kwargs) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self.survival_time_attr = kwargs.get("survival_time_attr")
        return BaseOperator.set_params(self, **kwargs)

    @staticmethod
    def _append_survival_time_columns(
        values, survival_time: Union[pd.Series, np.ndarray, list]
    ) -> Optional[str]:
        survival_time_attr: str = _DEFAULT_SURVIVAL_TIME_ATTR
        if isinstance(survival_time, pd.Series):
            if survival_time.name is None:
                survival_time.name = survival_time_attr
            else:
                survival_time_attr = survival_time.name
            values[survival_time.name] = survival_time
        elif isinstance(survival_time, np.ndarray):
            np.append(values, survival_time, axis=1)
        elif isinstance(survival_time, list):
            for index, row in enumerate(values):
                row.append(survival_time[index])
        else:
            raise ValueError(
                "Data values must be instance of either pandas DataFrame, numpy array"
                " or list"
            )
        return survival_time_attr

    def _prepare_survival_attribute(
        self, survival_time: Optional[Data], values: Data
    ) -> str:
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError(
                'No "survival_time" attribute name was specified. '
                + "Specify it using method set_params"
            )
        if survival_time is not None:
            return SurvivalRules._append_survival_time_columns(values, survival_time)
        return self.survival_time_attr

    def fit(
        self, values: Data, labels: Data, survival_time: Data = None
    ) -> SurvivalRules:  # pylint: disable=arguments-differ
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr*
            parameter was specified.

        Returns
        -------
        self : SurvivalRules
        """
        survival_time_attribute = self._prepare_survival_attribute(
            survival_time, values
        )
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
            Each row represent single example from dataset and contains estimated
            survival function for that example. Estimated survival function is returned
            as a dictionary containing times and corresponding probabilities.
        """
        return PredictionResultMapper.map_survival(super().predict(values))

    def score(self, values: Data, labels: Data, survival_time: Data = None) -> float:
        """Return the Integrated Brier Score on the given dataset and labels
         (event status indicator).

        Integrated Brier Score (IBS) - the Brier score (BS) represents the squared
        difference between true event status at time T and predicted event status at
        that time; the Integrated Brier score summarizes the prediction error over all
        observations and over all times in a test set.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr*
            parameter was specified

        Returns
        -------
        score : float
            Integrated Brier Score of self.predict(values) wrt. labels.
        """

        survival_time_attribute = self._prepare_survival_attribute(
            survival_time, values
        )
        example_set = ExampleSetFactory(self._get_problem_type()).make(
            values, labels, survival_time_attribute=survival_time_attribute
        )

        predicted_example_set = (
            self.model._java_object.apply(  # pylint: disable=protected-access
                example_set
            )
        )

        IntegratedBrierScore = JClass(  # pylint: disable=invalid-name
            "adaa.analytics.rules.logic.performance.IntegratedBrierScore"
        )
        integrated_brier_score = IntegratedBrierScore()
        ibs = integrated_brier_score.countExample(predicted_example_set).getValue()
        return float(ibs)

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.SURVIVAL


class ExpertSurvivalRules(ExpertKnowledgeOperator, SurvivalRules):
    """Expert Survival model."""

    __params_class__ = _SurvivalExpertModelParams

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments,too-many-locals
        self,
        survival_time_attr: str = None,
        minsupp_new: float = DEFAULT_PARAMS_VALUE["minsupp_new"],
        max_growing: int = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
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
        max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"],
    ):
        """
        Parameters
        ----------
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples to be covered by a new rule (positive examples for classification
            problems); default: 5,
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model
            is pandas dataframe).
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
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self._initialize_rulekit()
        self.set_params(
            survival_time_attr=survival_time_attr,
            minsupp_new=minsupp_new,
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
            max_rule_count=max_rule_count,
        )
        self.model: RuleSet[SurvivalRule] = None

    def set_params(self, **kwargs) -> object:  # pylint: disable=arguments-differ
        self.survival_time_attr = kwargs["survival_time_attr"]
        return ExpertKnowledgeOperator.set_params(self, **kwargs)

    def fit(  # pylint: disable=arguments-differ,too-many-arguments
        self,
        values: Data,
        labels: Data,
        survival_time: Data = None,
        expert_rules: list[Union[str, tuple[str, str]]] = None,
        expert_preferred_conditions: list[Union[str, tuple[str, str]]] = None,
        expert_forbidden_conditions: list[Union[str, tuple[str, str]]] = None,
    ) -> ExpertSurvivalRules:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : Data
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr*
            parameter was specified.
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
        self : ExpertSurvivalRules
        """
        survival_time_attribute = SurvivalRules._prepare_survival_attribute(
            self, survival_time, values
        )
        return ExpertKnowledgeOperator.fit(
            self,
            values=values,
            labels=labels,
            survival_time_attribute=survival_time_attribute,
            expert_rules=expert_rules,
            expert_preferred_conditions=expert_preferred_conditions,
            expert_forbidden_conditions=expert_forbidden_conditions,
        )

    def predict(self, values: Data) -> np.ndarray:
        return PredictionResultMapper.map_survival(
            ExpertKnowledgeOperator.predict(self, values)
        )

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.SURVIVAL


class _SurvivalContrastSetModelParams(ContrastSetModelParams, _SurvivalModelsParams):
    pass


class ContrastSetSurvivalRules(BaseOperator, _BaseSurvivalRulesModel):
    """Contrast set survival model."""

    __params_class__ = _SurvivalContrastSetModelParams

    def __init__(  # pylint: disable=super-init-not-called,too-many-arguments
        self,
        minsupp_all: Tuple[float, float, float, float] = DEFAULT_PARAMS_VALUE[
            "minsupp_all"
        ],
        max_neg2pos: float = DEFAULT_PARAMS_VALUE["max_neg2pos"],
        max_passes_count: int = DEFAULT_PARAMS_VALUE["max_passes_count"],
        penalty_strength: float = DEFAULT_PARAMS_VALUE["penalty_strength"],
        penalty_saturation: float = DEFAULT_PARAMS_VALUE["penalty_saturation"],
        survival_time_attr: str = None,
        minsupp_new: float = DEFAULT_PARAMS_VALUE["minsupp_new"],
        max_growing: int = DEFAULT_PARAMS_VALUE["max_growing"],
        enable_pruning: bool = DEFAULT_PARAMS_VALUE["enable_pruning"],
        ignore_missing: bool = DEFAULT_PARAMS_VALUE["ignore_missing"],
        max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE["max_uncovered_fraction"],
        select_best_candidate: bool = DEFAULT_PARAMS_VALUE["select_best_candidate"],
        complementary_conditions: bool = DEFAULT_PARAMS_VALUE[
            "complementary_conditions"
        ],
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
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model
            is pandas dataframe).
        minsupp_new : float = 5.0
            a minimum number (or fraction, if value < 1.0) of previously uncovered
            examples to be covered by a new rule (positive examples for classification
            problems); default: 5,
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
        max_rule_count : int = 0
            Maximum number of rules to be generated (for classification data sets it
            applies to a single class); 0 indicates no limit.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.contrast_attribute: str = None
        self._initialize_rulekit()
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
            max_rule_count=max_rule_count,
        )
        self.model: RuleSet[SurvivalRule] = None

    def set_params(self, **kwargs) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        # params validation
        self.survival_time_attr = kwargs["survival_time_attr"]
        return BaseOperator.set_params(self, **kwargs)

    def fit(  # pylint: disable=arguments-renamed
        self,
        values: Data,
        labels: Data,
        contrast_attribute: str,
        survival_time: Data = None,
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
            data about survival time. Could be omitted when *survival_time_attr*
            parameter was specified.

        Returns
        -------
        self : ContrastSetSurvivalRules
        """
        survival_time_attribute = SurvivalRules._prepare_survival_attribute(  # pylint: disable=protected-access
            self, survival_time, values
        )
        super().fit(
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute,
        )
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
            Each row represent single example from dataset and contains estimated
            survival function for that example. Estimated survival function is returned
            as a dictionary containing times and corresponding probabilities.
        """
        return PredictionResultMapper.map_survival(super().predict(values))

    def score(self, values: Data, labels: Data, survival_time: Data = None) -> float:
        """Return the Integrated Brier Score on the given dataset and
         labels(event status indicator).

        Integrated Brier Score (IBS) - the Brier score (BS) represents the squared
        differencebetween true event status at time T and predicted event status at that
        time; the Integrated Brier score summarizes the prediction error over all
        observations and over all times in a test set.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr*
            parameter was specified

        Returns
        -------
        score : float
            Integrated Brier Score of self.predict(values) wrt. labels.
        """
        return SurvivalRules.score(self, values, labels, survival_time=survival_time)

    def _get_problem_type(self) -> ProblemType:
        return ProblemType.CONTRAST_SURVIVAL
