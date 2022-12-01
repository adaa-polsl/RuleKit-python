from .helpers import RuleGeneratorConfigurator, \
    PredictionResultMapper, \
    create_example_set, \
    get_rule_generator, \
    ModelSerializer
from .params import Measures, ModelsParams
from .rules import RuleSet, Rule
import numpy as np
import pandas as pd
from typing import Union, Dict, Any, List

Data = Union[np.ndarray, pd.DataFrame, List]

DEFAULT_PARAMS_VALUE = {
    'minsupp_new': 5,
    'min_rule_covered': 5,
    'induction_measure': Measures.Correlation,
    'pruning_measure':  Measures.Correlation,
    'voting_measure': Measures.Correlation,
    'max_growing': 0.0,
    'enable_pruning': True,
    'ignore_missing': False,
    'max_uncovered_fraction': 0.0,
    'select_best_candidate': False,

    'extend_using_preferred': None,
    'extend_using_automatic': None,
    'induce_using_preferred': None,
    'induce_using_automatic': None,
    'consider_other_classes': None,
    'preferred_conditions_per_rule': None,
    'preferred_attributes_per_rule': None,
}


class BaseOperator:

    def __init__(self, **kwargs):
        self._params = None
        self._rule_generator = None
        self.set_params(**kwargs)
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

    def get_params(self, deep=True) -> Dict[str, Any]:
        """
        Returns
        -------
        hyperparameters : np.ndarray
            Dictionary containing model hyperparameters.
        """
        return self._params

    def set_params(self, **kwargs) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self._rule_generator = get_rule_generator()
        # validate
        ModelsParams(**kwargs)
        self._params = kwargs
        configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = configurator.configure(**kwargs)
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

    def __init__(self, **kwargs):
        self._params = None
        self._rule_generator = None
        self._configurator = None
        ExpertKnowledgeOperator.set_params(self, **kwargs)
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

    def set_params(self, **kwargs) -> object:
        # validate params
        ModelsParams(**kwargs)
        self._params = kwargs
        self._rule_generator = get_rule_generator(expert=True)
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = self._configurator.configure(**kwargs)
        return self
