from .helpers import RuleGeneratorConfigurator, \
    PredictionResultMapper, \
    create_example_set, \
    get_rule_generator, \
    create_sorted_example_set, \
    ModelSerializer
from .params import Measures
from .rules import RuleSet, Rule
import numpy as np
import pandas as pd
from typing import Union, Any, List

Data = Union[np.ndarray, pd.DataFrame, List]


class BaseOperator:

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        self._params = None
        self._rule_generator = None
        self.set_params(
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing
        )
        self.model: RuleSet = None

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map(predicted_example_set)

    def fit(self, values: Data, labels: Data, survival_time_attribute: str = None) -> Any:
        example_set = create_example_set(
            values, labels, survival_time_attribute=survival_time_attribute)
        java_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(java_model)
        return self.model

    def predict(self, values: Data) -> np.ndarray:
        if self.model is None:
            raise ValueError(
                '"fit" method must be called before calling this method')
        example_set = create_example_set(values)
        return self.model._java_object.apply(example_set)

    def get_params(self, deep=True) -> dict:
        """
        Returns
        -------
        hyperparameters : np.ndarray
            Dictionary containing model hyperparameters.
        """
        return self._params

    def set_params(self,
                   min_rule_covered: int = None,
                   induction_measure: Measures = None,
                   pruning_measure: Union[Measures, str] = None,
                   voting_measure: Measures = None,
                   max_growing: int = None,
                   enable_pruning: bool = None,
                   ignore_missing: bool = None) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self._rule_generator = get_rule_generator()
        self._params = dict(
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
        )
        configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = configurator.configure(**self._params)
        return self

    def get_coverage_matrix(self, values: Data) -> np.ndarray:
        """Calculates coverage matrix for ruleset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            dataset

        Returns
        -------
        coverage_matrix : np.ndarray
             Each row of the matrix represent single example from dataset and every column represent
            on rule from rule set. Value 1 in the matrix cell means that rule covered certain example, value 0
            means that it doesn't.
        """
        if self.model is None:
            raise ValueError(
                '"fit" method must be called before calling this method')
        covering_info = self.model.covering(create_example_set(values))
        if isinstance(values, pd.Series) or isinstance(values, pd.DataFrame):
            values = values.to_numpy()
        result = []
        i = 0
        for row in values:
            row_result = []
            for item in covering_info:
                value = 0 if item is None or not i in item else 1
                row_result.append(value)
            result.append(np.array(row_result))
            i += 1
        return np.array(result)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop('_rule_generator')
        return {
            '_params': self._params,
            'model': ModelSerializer.serialize(self.model)
        }

    def __setstate__(self, state: dict):
        self.model = ModelSerializer.deserialize(state['model'])
        self._rule_generator = get_rule_generator()
        self.set_params(**state['_params'])


class ExpertKnowledgeOperator:

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
        self._params = None
        self._rule_generator = None
        self._configurator = None
        ExpertKnowledgeOperator.set_params(
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
        self.model: RuleSet = None

    def fit(self,
            values: Data,
            labels: Data,
            survival_time_attribute: str = None,

            expert_rules: List[Union[str, Rule]] = None,
            expert_preferred_conditions: List[Union[str, Rule]] = None,
            expert_forbidden_conditions: List[Union[str, Rule]] = None) -> Any:
        self._configurator.configure_simple_parameter('use_expert', True)
        self._configurator.configure_expert_parameter(
            'expert_preferred_conditions', expert_preferred_conditions)
        self._configurator.configure_expert_parameter(
            'expert_forbidden_conditions', expert_forbidden_conditions)
        self._configurator.configure_expert_parameter(
            'expert_rules', expert_rules)
        example_set = create_example_set(
            values, labels, survival_time_attribute=survival_time_attribute)

        java_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(java_model)
        return self.model

    def predict(self, values: Data) -> np.ndarray:
        if self.model is None:
            raise ValueError(
                '"fit" method must be called before calling this method')
        example_set = create_example_set(values)
        return self.model._java_object.apply(example_set)

    def set_params(self,
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
                   preferred_attributes_per_rule: int = None) -> object:
        self._params = dict(
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
            preferred_attributes_per_rule=preferred_attributes_per_rule,
        )
        self._rule_generator = get_rule_generator(expert=True)
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = self._configurator.configure(**self._params)
        return self
