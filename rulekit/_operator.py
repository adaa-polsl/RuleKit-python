"""Contains base classes for rule induction operators
"""
from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ._helpers import (ModelSerializer, PredictionResultMapper,
                       RuleGeneratorConfigurator, create_example_set,
                       get_rule_generator)
from .events import RuleInductionProgressListener, command_proxy_client_factory
from .main import RuleKit
from .rules import Rule, RuleSet

Data = Union[np.ndarray, pd.DataFrame, list]


class BaseOperator:
    """Base class for rule induction operator
    """

    __params_class__: type = None

    def __init__(self, **kwargs):
        self._initialize_rulekit()
        self._params = None
        self._rule_generator = None
        self.set_params(**kwargs)
        self.model: RuleSet = None

    def _initialize_rulekit(self):
        if not RuleKit.initialized:
            RuleKit.init()

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map(predicted_example_set)

    def _validate_contrast_attribute(
        self,
        example_set,
        contrast_attribute: Optional[str]
    ) -> None:
        contrast_attribute_instance = example_set.getAttributes().get(contrast_attribute)
        if contrast_attribute is not None and contrast_attribute_instance.isNumerical():
            raise ValueError(
                'Contrast set attributes must be a nominal attribute while ' +
                f'"{contrast_attribute}" is a numerical one.'
            )

    def _sanitize_parameters(self, params: dict[str, Any]) -> None:
        if params.get('minsupp_all', None) is not None and len(params['minsupp_all']) > 0:
            params['minsupp_all'] = ' '.join(
                [str(e) for e in params['minsupp_all']]
            )

    def fit(  # pylint: disable=missing-function-docstring
        self,
        values: Data,
        labels: Data,
        survival_time_attribute: str = None,
        contrast_attribute: str = None,
    ) -> BaseOperator:
        example_set = create_example_set(
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute
        )
        self._validate_contrast_attribute(example_set, contrast_attribute)

        java_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(java_model)
        return self.model

    def predict(self, values: Data) -> np.ndarray:  # pylint: disable=missing-function-docstring
        if self.model is None:
            raise ValueError(
                '"fit" method must be called before calling this method')
        example_set = create_example_set(values)
        return self.model._java_object.apply(example_set)  # pylint: disable=protected-access

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # pylint: disable=unused-argument
        """
        Parameters
        ----------
        deep : :class:`rulekit.operator.Data`
            Parameter for scikit-learn compatibility. Not used.

        Returns
        -------
        hyperparameters : np.ndarray
            Dictionary containing model hyperparameters.
        """
        return self._params

    def set_params(self, **kwargs) -> object:
        """Set models hyperparameters. Parameters are the same as in constructor."""
        self._rule_generator = self._get_rule_generator()
        self._sanitize_parameters(kwargs)
        params: BaseModel = self.__params_class__( # pylint: disable=not-callable
            **kwargs)  
        params_dict: dict = params.model_dump()
        self._params = {
            key: value for key, value in params_dict.items()
            if value is not None
        }
        configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = configurator.configure(**params_dict)
        return self

    def get_metadata_routing(self) -> None:
        """
        .. warning:: Scikit-learn metadata routing is not supported yet.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(
            'Scikit-learn metadata routing is not supported yet.'
        )

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
            on rule from rule set. Value 1 in the matrix cell means that rule covered certain 
            example, value 0 means that it doesn't.
        """
        if self.model is None:
            raise ValueError(
                '"fit" method must be called before calling this method')
        covering_info = self.model.covering(create_example_set(values))
        if isinstance(values, pd.Series) or isinstance(values, pd.DataFrame):
            values = values.to_numpy()
        result = []
        for i in range(len(values)):
            row_result: list[int] = []
            for item in covering_info:
                value = 0 if item is None or not i in item else 1
                row_result.append(value)
            result.append(np.array(row_result))
        return np.array(result)

    def add_event_listener(self, listener: RuleInductionProgressListener):
        command_proxy = command_proxy_client_factory(listener)
        self._rule_generator.addOperatorCommandProxy(command_proxy)

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

    def _get_rule_generator(self) -> RuleGeneratorConfigurator:
        return get_rule_generator()


class ExpertKnowledgeOperator(BaseOperator):
    """Base class for expert rule induction operator
    """

    def fit(  # pylint: disable=missing-function-docstring
        self,
        values: Data,
        labels: Data,
        survival_time_attribute: str = None,
        contrast_attribute: str = None,

        expert_rules: list[Union[str, Rule]] = None,
        expert_preferred_conditions: list[Union[str, Rule]] = None,
        expert_forbidden_conditions: list[Union[str, Rule]] = None
    ) -> ExpertKnowledgeOperator:
        example_set = create_example_set(
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute
        )
        self._validate_contrast_attribute(example_set, contrast_attribute)
        self._configure_expert_parameters(
            expert_rules,
            expert_preferred_conditions,
            expert_forbidden_conditions,
        )
        java_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(java_model)
        return self.model

    def _get_rule_generator(self) -> RuleGeneratorConfigurator:
        return get_rule_generator(expert=True)
    
    def _configure_expert_parameters(
        self,
        expert_rules: list[Union[str, Rule]] = None,
        expert_preferred_conditions: list[Union[str, Rule]] = None,
        expert_forbidden_conditions: list[Union[str, Rule]] = None
    ) -> None:
        configurator = RuleGeneratorConfigurator(self._rule_generator)
        configurator._configure_simple_parameter(  # pylint: disable=protected-access
            'use_expert', True)
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            'expert_preferred_conditions',
            self._sanitize_expert_parameter(expert_preferred_conditions)
        )
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            'expert_forbidden_conditions',
            self._sanitize_expert_parameter(expert_forbidden_conditions)
        )
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            'expert_rules',
            self._sanitize_expert_parameter(expert_rules)
        )

    def _sanitize_expert_parameter(
        self,
        expert_parameter: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        if expert_parameter is None:
            return None
        sanitized_parameter: list[tuple[str, str]] = []
        for item in expert_parameter:
            item_id, item_value = item
            sanitized_parameter.append((item_id, item_value))
        return sanitized_parameter
