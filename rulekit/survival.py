from typing import Any, Union, Tuple, List
import numpy as np
import pandas as pd
from jpype import JClass

from .helpers import PredictionResultMapper, RuleGeneratorConfigurator, get_rule_generator, create_example_set
from .operator import BaseOperator, ExpertKnowledgeOperator, Data, DEFAULT_PARAMS_VALUE
from .rules import RuleSet



class SurvivalRules(BaseOperator):
    """Survival model."""

    def __init__(self,
                 survival_time_attr: str = None,
                 min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                 max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
                 enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                 ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                 max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                 select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate']):
        """
        Parameters
        ----------
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model is padnas dataframe).
        min_rule_covered : int = 5
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule 
            (positive examples for classification problems); default: 5
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase 
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always 
            considered as not fulfilling the condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default: False.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.set_params(
            survival_time_attr=survival_time_attr,
            min_rule_covered=min_rule_covered,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate)
        self.model: RuleSet = None

    def set_params(self,
                   survival_time_attr: str = None,
                   min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                   max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
                   enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                   ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                   max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                   select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate']) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self.survival_time_attr = survival_time_attr
        self._rule_generator = get_rule_generator()
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._params = dict(
            survival_time_attr=survival_time_attr,
            min_rule_covered=min_rule_covered,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            max_uncovered_fraction=max_uncovered_fraction,
            select_best_candidate=select_best_candidate
        )
        self._rule_generator = self._configurator.configure(**self._params)
        return self

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
                'Data values must be instance of either pandas DataFrame, numpy array or list')
        return ''

    def fit(self, values: Data, labels: Data, survival_time: Data = None) -> Any:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was specified.

        Returns
        -------
        self : SurvivalRules
        """
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError(
                'No "survival_time" attribute name was specified. Specify it using method set_params')
        if survival_time is not None:
            survival_time_attribute = SurvivalRules._append_survival_time_columns(
                values, survival_time)
        else:
            survival_time_attribute = self.survival_time_attr
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
            Each row represent single example from dataset and contains estimated survival function for that example. Estimated survival function is returned as a dictionary containing times and corresponding probabilities.
        """
        return PredictionResultMapper.map_survival(super().predict(values))

    def score(self, values: Data, labels: Data, survival_time: Data = None) -> float:
        """Return the Integrated Brier Score on the given dataset and labels(event status indicator).

        Integrated Brier Score (IBS) - the Brier score (BS) represents the squared difference between true event status at time T and predicted event status at that time; 
        the Integrated Brier score summarizes the prediction error over all observations and over all times in a test set.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : :class:`rulekit.operator.Data`
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was specified

        Returns
        -------
        score : float
            Integrated Brier Score of self.predict(values) wrt. labels.
        """

        if self.survival_time_attr is None and survival_time is None:
            raise ValueError(
                'No "survival_time" attribute name was specified. Specify it using method set_params')
        if survival_time is not None:
            survival_time_attribute = SurvivalRules._append_survival_time_columns(
                values, survival_time)
        else:
            survival_time_attribute = self.survival_time_attr

        example_set = create_example_set(
            values, labels,  survival_time_attribute=survival_time_attribute)

        predicted_example_set = self.model._java_object.apply(example_set)

        IntegratedBrierScore = JClass(
            'adaa.analytics.rules.logic.quality.IntegratedBrierScore')
        integratedBrierScore = IntegratedBrierScore()
        integratedBrierScore.startCounting(predicted_example_set, True)
        return integratedBrierScore.getMikroAverage()


class ExpertSurvivalRules(ExpertKnowledgeOperator, SurvivalRules):
    """Expert Survival model."""

    def __init__(self,
                 survival_time_attr: str = None,
                 min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                 max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
                 enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                 ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                 max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                 select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],

                 extend_using_preferred: bool = DEFAULT_PARAMS_VALUE['extend_using_preferred'],
                 extend_using_automatic: bool = DEFAULT_PARAMS_VALUE['extend_using_automatic'],
                 induce_using_preferred: bool = DEFAULT_PARAMS_VALUE['induce_using_preferred'],
                 induce_using_automatic: bool = DEFAULT_PARAMS_VALUE['induce_using_automatic'],
                 preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
                     'preferred_conditions_per_rule'],
                 preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE['preferred_attributes_per_rule']):
        """
        Parameters
        ----------
        survival_time_attr : str
            name of column containing survival time data (use when data passed to model is padnas dataframe).
        min_rule_covered : int = 5
            positive integer representing minimum number of previously uncovered examples to be covered by a new rule 
            (positive examples for classification problems); default: 5
        max_growing : int = 0.0
            non-negative integer representing maximum number of conditions which can be added to the rule in the growing phase 
            (use this parameter for large datasets if execution time is prohibitive); 0 indicates no limit; default: 0,
        enable_pruning : bool = True
            enable or disable pruning, default is True.
        ignore_missing : bool = False
            boolean telling whether missing values should be ignored (by default, a missing value of given attribute is always 
            considered as not fulfilling the condition build upon that attribute); default: False.
        max_uncovered_fraction : float = 0.0
            Floating-point number from [0,1] interval representing maximum fraction of examples that may remain uncovered by the rule set, default: 0.0.
        select_best_candidate : bool = False
            Flag determining if best candidate should be selected from growing phase; default: False.

        extend_using_preferred : bool = False
            boolean indicating whether initial rules should be extended with a use of preferred conditions and attributes; default is False
        extend_using_automatic : bool = False
            boolean indicating whether initial rules should be extended with a use of automatic conditions and attributes; default is False
        induce_using_preferred : bool = False
            boolean indicating whether new rules should be induced with a use of preferred conditions and attributes; default is False
        induce_using_automatic : bool = False
            boolean indicating whether new rules should be induced with a use of automatic conditions and attributes; default is False
        preferred_conditions_per_rule : int = None
            maximum number of preferred conditions per rule; default: unlimited,
        preferred_attributes_per_rule : int = None
            maximum number of preferred attributes per rule; default: unlimited.
        """
        self._params = None
        self._rule_generator = None
        self._configurator = None
        self.set_params(
            survival_time_attr=survival_time_attr,
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
            preferred_attributes_per_rule=preferred_attributes_per_rule
        )
        self.model: RuleSet = None

    def set_params(self,
                   survival_time_attr: str = None,
                   min_rule_covered: int = DEFAULT_PARAMS_VALUE['min_rule_covered'],
                   max_growing: int = DEFAULT_PARAMS_VALUE['max_growing'],
                   enable_pruning: bool = DEFAULT_PARAMS_VALUE['enable_pruning'],
                   ignore_missing: bool = DEFAULT_PARAMS_VALUE['ignore_missing'],
                   max_uncovered_fraction: float = DEFAULT_PARAMS_VALUE['max_uncovered_fraction'],
                   select_best_candidate: bool = DEFAULT_PARAMS_VALUE['select_best_candidate'],

                   extend_using_preferred: bool = DEFAULT_PARAMS_VALUE['extend_using_preferred'],
                   extend_using_automatic: bool = DEFAULT_PARAMS_VALUE['extend_using_automatic'],
                   induce_using_preferred: bool = DEFAULT_PARAMS_VALUE['induce_using_preferred'],
                   induce_using_automatic: bool = DEFAULT_PARAMS_VALUE['induce_using_automatic'],
                   preferred_conditions_per_rule: int = DEFAULT_PARAMS_VALUE[
                       'preferred_conditions_per_rule'],
                   preferred_attributes_per_rule: int = DEFAULT_PARAMS_VALUE['preferred_attributes_per_rule']) -> object:

        self.survival_time_attr = survival_time_attr

        self._params = dict(
            survival_time_attr=survival_time_attr,
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
        )
        self._rule_generator = get_rule_generator(expert=True)
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = self._configurator.configure(**self._params)
        return self

    def fit(self,
            values: Data,
            labels: Data,
            survival_time: Data = None,

            expert_rules: List[Union[str, Tuple[str, str]]] = None,
            expert_preferred_conditions: List[Union[str,
                                                    Tuple[str, str]]] = None,
            expert_forbidden_conditions: List[Union[str, Tuple[str, str]]] = None) -> Any:
        """Train model on given dataset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            attributes
        labels : Data
            survival status
        survival_time: :class:`rulekit.operator.Data`
            data about survival time. Could be omitted when *survival_time_attr* parameter was specified.

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
        self : ExpertSurvivalRules
        """
        if self.survival_time_attr is None and survival_time is None:
            raise ValueError('No "survival_time" attribute name was specified. '
                             'Specify it or pass its values by "survival_time" parameter.')
        if survival_time is not None:
            survival_time_attribute = SurvivalRules._append_survival_time_columns(
                values, survival_time)
        else:
            survival_time_attribute = self.survival_time_attr
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
