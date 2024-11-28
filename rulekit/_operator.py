"""Contains base classes for rule induction operators
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from rulekit._helpers import ExampleSetFactory
from rulekit._helpers import get_rule_generator
from rulekit._helpers import ModelSerializer
from rulekit._helpers import PredictionResultMapper
from rulekit._helpers import RuleGeneratorConfigurator
from rulekit._problem_types import ProblemType
from rulekit.events import _command_listener_factory
from rulekit.events import RuleInductionProgressListener
from rulekit.main import RuleKit
from rulekit.rules import BaseRule
from rulekit.rules import RuleSet

Data = Union[np.ndarray, pd.DataFrame, list]


class BaseOperator(ABC):
    """Base class for rule induction operator"""

    __params_class__: type = None

    def __init__(self, **kwargs):
        self._initialize_rulekit()
        self._params = None
        self._rule_generator = None
        self.set_params(**kwargs)
        self.model: RuleSet[BaseRule] = None

    def _initialize_rulekit(self):
        if not RuleKit.initialized:
            RuleKit.init()

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map(predicted_example_set)

    def _validate_contrast_attribute(
        self, example_set, contrast_attribute: Optional[str]
    ) -> None:
        if contrast_attribute is None:
            return
        contrast_attribute_instance = example_set.getAttributes().get(
            contrast_attribute
        )
        if contrast_attribute_instance.isNumerical():
            raise ValueError(
                "Contrast set attributes must be a nominal attribute while "
                + f'"{contrast_attribute}" is a numerical one.'
            )

    def fit(  # pylint: disable=missing-function-docstring
        self,
        values: Data,
        labels: Data,
        survival_time_attribute: str = None,
        contrast_attribute: str = None,
    ) -> BaseOperator:
        example_set = ExampleSetFactory(self._get_problem_type()).make(
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute,
        )
        self._validate_contrast_attribute(example_set, contrast_attribute)

        java_model = self._rule_generator.learn(example_set)
        self.model = RuleSet[BaseRule](java_model)
        return self.model

    def predict(
        self, values: Data
    ) -> np.ndarray:  # pylint: disable=missing-function-docstring
        if self.model is None:
            raise ValueError('"fit" method must be called before calling this method')
        example_set = ExampleSetFactory(self._get_problem_type()).make(values)
        return self.model._java_object.apply(  # pylint: disable=protected-access
            example_set
        )

    def get_params(
        self, deep: bool = True  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
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
        params: BaseModel = self.__params_class__(  # pylint: disable=not-callable
            **kwargs
        )
        params_dict: dict = params.model_dump()
        self._params = {
            key: value for key, value in params_dict.items() if value is not None
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
        raise NotImplementedError("Scikit-learn metadata routing is not supported yet.")

    def get_coverage_matrix(self, values: Data) -> np.ndarray:
        """Calculates coverage matrix for ruleset.

        Parameters
        ----------
        values : :class:`rulekit.operator.Data`
            dataset

        Returns
        -------
        coverage_matrix : np.ndarray
            Each row of the matrix represent single example from dataset and every
            column represent on rule from rule set. Value 1 in the matrix cell means
            that rule covered certain example, value 0 means that it doesn't.
        """
        if self.model is None:
            raise ValueError('"fit" method must be called before calling this method')
        example_set = ExampleSetFactory(self._get_problem_type()).make(values)
        covering_info = self.model.covering(example_set)
        if isinstance(values, (pd.Series, pd.DataFrame)):
            values = values.to_numpy()
        result = []
        for i in range(len(values)):
            row_result: list[int] = [
                0 if item is None or i not in item else 1 for item in covering_info
            ]
            result.append(np.array(row_result))
        return np.array(result)

    def add_event_listener(self, listener: RuleInductionProgressListener):
        """Add event listener object to the operator which allows to monitor
         rule induction progress.

        Example:
            >>> from rulekit.events import RuleInductionProgressListener
            >>> from rulekit.classification import RuleClassifier
            >>>
            >>> class MyEventListener(RuleInductionProgressListener):
            >>>     def on_new_rule(self, rule):
            >>>         print('Do something with new rule', rule)
            >>>
            >>> operator = RuleClassifier()
            >>> operator.add_event_listener(MyEventListener())

        Args:
            listener (RuleInductionProgressListener): listener object
        """
        command_listener = _command_listener_factory(listener)
        self._rule_generator.addOperatorListener(command_listener)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_rule_generator")
        return {"_params": self._params, "model": ModelSerializer.serialize(self.model)}

    def __setstate__(self, state: dict):
        self.model = ModelSerializer.deserialize(state["model"])
        self._rule_generator = get_rule_generator()
        self.set_params(**state["_params"])

    def _get_rule_generator(self) -> RuleGeneratorConfigurator:
        return get_rule_generator()

    @abstractmethod
    def _get_problem_type(self) -> ProblemType:
        pass


class ExpertKnowledgeOperator(BaseOperator, ABC):
    """Base class for expert rule induction operator"""

    def fit(  # pylint: disable=missing-function-docstring,too-many-arguments
        self,
        values: Data,
        labels: Data,
        survival_time_attribute: str = None,
        contrast_attribute: str = None,
        expert_rules: list[Union[str, BaseRule]] = None,
        expert_preferred_conditions: list[Union[str, BaseRule]] = None,
        expert_forbidden_conditions: list[Union[str, BaseRule]] = None,
    ) -> ExpertKnowledgeOperator:
        example_set = ExampleSetFactory(self._get_problem_type()).make(
            values,
            labels,
            survival_time_attribute=survival_time_attribute,
            contrast_attribute=contrast_attribute,
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
        expert_rules: Optional[list[Union[str, BaseRule]]] = None,
        expert_preferred_conditions: Optional[list[Union[str, BaseRule]]] = None,
        expert_forbidden_conditions: Optional[list[Union[str, BaseRule]]] = None,
    ) -> None:
        if expert_rules is None:
            expert_rules = []
        if expert_preferred_conditions is None:
            expert_preferred_conditions = []
        if expert_forbidden_conditions is None:
            expert_forbidden_conditions = []

        configurator = RuleGeneratorConfigurator(self._rule_generator)
        configurator._configure_simple_parameter(  # pylint: disable=protected-access
            "use_expert", True
        )
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            "expert_preferred_conditions",
            self._sanitize_expert_parameter(expert_preferred_conditions),
        )
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            "expert_forbidden_conditions",
            self._sanitize_expert_parameter(expert_forbidden_conditions),
        )
        configurator._configure_expert_parameter(  # pylint: disable=protected-access
            "expert_rules", self._sanitize_expert_parameter(expert_rules)
        )

    def _sanitize_expert_parameter(
        self, expert_parameter: Optional[list[tuple[str, str]]]
    ) -> list[tuple[str, str]]:
        if expert_parameter is None:
            return None
        sanitized_parameter: list[tuple[str, str]] = []
        for item in expert_parameter:
            item_id, item_value = item
            sanitized_parameter.append((item_id, item_value))
        return sanitized_parameter
