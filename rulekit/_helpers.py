"""Contains helper functions and classes
"""
import io
import json
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from jpype import JArray
from jpype import java
from jpype import JClass
from jpype import JObject
from jpype.pickle import JPickler
from jpype.pickle import JUnpickler

from rulekit._logging import _RuleKitJavaLoggerConfig
from rulekit._problem_types import ProblemType
from rulekit.exceptions import RuleKitMisconfigurationException
from rulekit.main import RuleKit
from rulekit.params import Measures
from rulekit.rules import BaseRule


def get_rule_generator(expert: bool = False) -> Any:
    """Factory for Java RuleGenerator class object

    Args:
        expert (bool, optional): Whether expert induction is enables.
         Defaults to False.

    Returns:
        Any: RuleGenerator instance
    """
    RuleGenerator = JClass(  # pylint: disable=invalid-name
        "adaa.analytics.rules.logic.rulegenerator.RuleGenerator"
    )
    rule_generator = RuleGenerator(expert)
    return rule_generator


class RuleGeneratorConfigurator:
    """Class for configuring rule induction parameters"""

    _MEASURES_PARAMETERS: list[str] = [
        "induction_measure",
        "pruning_measure",
        "voting_measure",
    ]
    _USER_DEFINED_MEASURE_VALUE: str = "UserDefined"

    def __init__(self, rule_generator):
        self.rule_generator = rule_generator
        self.LogRank = None  # pylint: disable=invalid-name

    def configure(self, **params: dict[str, Any]) -> Any:
        """Configures RuleGenerator instance with given induction parameters

        Returns:
            Any: configured RuleGenerator instance
        """
        self._configure_rule_generator(**params)
        self._validate_rule_generator_parameters(**params)
        self._configure_java_logging()
        return self.rule_generator

    def _configure_expert_parameter(self, param_name: str, param_value: Any):
        if param_value is None:
            return
        rules_list = java.util.ArrayList()
        if isinstance(param_value, list) and len(param_value) > 0:
            if isinstance(param_value[0], str):
                for index, rule in enumerate(param_value):
                    rule_name = f"{param_name[:-1]}-{index}"
                    rules_list.add(
                        JObject([rule_name, rule], JArray("java.lang.String", 1))
                    )
            elif isinstance(param_value[0], BaseRule):
                for index, rule in enumerate(param_value):
                    rule_name = f"{param_name[:-1]}-{index}"
                    rules_list.add(
                        JObject([rule_name, str(rule)], JArray("java.lang.String", 1))
                    )
            elif isinstance(param_value[0], tuple):
                for index, rule in enumerate(param_value):
                    rules_list.add(
                        JObject([rule[0], rule[1]], JArray("java.lang.String", 1))
                    )
        self.rule_generator.setListParameter(param_name, rules_list)

    def _configure_simple_parameter(self, param_name: str, param_value: Any):
        if param_value is not None:
            if isinstance(param_value, bool):
                param_value = (str(param_value)).lower()
            elif isinstance(param_value, tuple):
                param_value = " ".join(list(map(str, param_value)))
            elif not isinstance(param_value, str):
                param_value = str(param_value)
            self.rule_generator.setParameter(param_name, param_value)

    def _configure_measure_parameter(
        self, param_name: str, param_value: Union[str, Measures]
    ):
        if param_value is not None:
            if isinstance(param_value, Measures):
                self.rule_generator.setParameter(param_name, param_value.value)
            if isinstance(param_value, Callable):
                self._configure_user_defined_measure_parameter(param_name, param_value)

    def _configure_user_defined_measure_parameter(
        self, param_name: str, param_value: Any
    ):
        from rulekit.params import _user_defined_measure_factory

        user_defined_measure = _user_defined_measure_factory(param_value)
        {
            "induction_measure": self.rule_generator.setUserMeasureInductionObject,
            "pruning_measure": self.rule_generator.setUserMeasurePurningObject,
            "voting_measure": self.rule_generator.setUserMeasureVotingObject,
        }[param_name](user_defined_measure)
        self.rule_generator.setParameter(param_name, self._USER_DEFINED_MEASURE_VALUE)

    def _configure_rule_generator(self, **kwargs: dict[str, Any]):
        if any(
            [
                kwargs.get(param_name) == Measures.LogRank
                for param_name in self._MEASURES_PARAMETERS
            ]
        ):
            self.LogRank = JClass("adaa.analytics.rules.logic.quality.LogRank")
        for measure_param_name in self._MEASURES_PARAMETERS:
            measure_param_value: Measures = kwargs.pop(measure_param_name, None)
            self._configure_measure_parameter(measure_param_name, measure_param_value)
        for param_name, param_value in kwargs.items():
            self._configure_simple_parameter(param_name, param_value)

    def _validate_rule_generator_parameters(self, **python_parameters: dict[str, Any]):
        """Validate whether operator parameters configuration is properly passed to the
        java RuleGenerator class. Otherwise, it raises an error.

        Args:
            python_parameters (dict[str, Any]): parameters values configured for the
                operator in Python

        Raises:
            ValueError: If failed to retrieve RuleGenerator parameters JSON
            RuleKitMisconfigurationException: If Java and Python parameters do not match
        """

        def are_params_equal(
            java_params: dict[str, Any], python_params: dict[str, Any]
        ):
            if java_params.keys() != python_params.keys():
                return False
            for key in java_params.keys():
                java_value = java_params.get(key)
                python_value = python_params.get(key)
                skip_check: bool = isinstance(python_value, Callable)
                if java_value == 'None':
                    java_value = None
                if java_value is None and python_value is None:
                    continue
                if java_value != python_value and not skip_check:
                    return False
            return True

        python_parameters = dict(python_parameters)
        for param_name, param_value in python_parameters.items():
            # convert measures to strings values for comparison
            if isinstance(param_value, Measures):
                python_parameters[param_name] = param_value.value
            # convert booleans to lowercase strings for comparison
            elif isinstance(param_value, bool):
                python_parameters[param_name] = str(param_value).lower()
            # normalize tuples to strings for comparison
            elif isinstance(param_value, tuple):
                value = " ".join(list(map(str, param_value)))
                python_parameters[param_name] = value
            # convert numbers to strings for comparison
            elif isinstance(param_value, (int, float)):
                python_parameters[param_name] = str(param_value)
        java_params_json: str = str(self.rule_generator.getParamsAsJsonString())
        try:
            java_params: dict[str, Any] = json.loads(java_params_json)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Failed to decode RuleGenerator parameters JSON"
            ) from error
        # select only values that are used by Python wrapper
        java_params = {
            param_name: str(java_params.get(param_name))
            for param_name in python_parameters.keys()
        }
        if not are_params_equal(java_params, python_parameters):
            raise RuleKitMisconfigurationException(
                java_parameters=java_params, python_parameters=python_parameters
            )

    def _configure_java_logging(self):
        logger_config: Optional[_RuleKitJavaLoggerConfig] = (
            RuleKit.get_java_logger_config()
        )
        if logger_config is None:
            return
        self.rule_generator.configureLogger(
            logger_config.log_file_path, logger_config.verbosity_level
        )


class ExampleSetFactory:
    """Creates ExampleSet object from given data"""

    DEFAULT_LABEL_ATTRIBUTE_NAME: str = "label"
    AUTOMATIC_ATTRIBUTES_NAMES_PREFIX: str = "att"

    def __init__(self, problem_type: ProblemType) -> None:
        self._problem_type: ProblemType = problem_type
        self._attributes_names: list[str] = None
        self._label_name: str = None
        self._survival_time_attribute: str = None
        self._contrast_attribute: str = None
        self._X: np.ndarray = None
        self._y: np.ndarray = None

    def make(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        survival_time_attribute: str = None,
        contrast_attribute: str = None,
    ) -> JObject:
        """Creates ExampleSet object from given data

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Data
            y (Union[pd.Series, np.ndarray], optional): Labels. Defaults to None.
                survival_time_attribute (str, optional): Name of survival time
                attribute. Defaults to None.
            contrast_attribute (str, optional): Name of contrast attribute.
                Defaults to None.

        Returns:
            JObject: ExampleSet object
        """
        self._attributes_names = []
        self._survival_time_attribute = survival_time_attribute
        self._contrast_attribute = contrast_attribute
        self._sanitize_X(X)
        self._sanitize_y(y)
        self._validate_X()
        self._validate_y()
        return self._create_example_set()

    def _sanitize_y(self, y: Union[pd.Series, np.ndarray, list]):
        if y is None:
            return
        elif isinstance(y, pd.Series):
            self._label_name = y.name
            self._attributes_names.append(self._label_name)
            self._y = y.to_numpy()
        elif isinstance(y, list):
            self._label_name = self.DEFAULT_LABEL_ATTRIBUTE_NAME
            self._attributes_names.append(self._label_name)
            self._y = np.array(y)
        elif isinstance(y, np.ndarray):
            self._label_name = self.DEFAULT_LABEL_ATTRIBUTE_NAME
            self._attributes_names.append(self._label_name)
            self._y = y
        else:
            raise ValueError(
                f"Invalid y type: {str(type(y))}. "
                "Supported types are: 1 dimensional numpy array or pandas "
                "Series object."
            )

    def _sanitize_X(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            self._attributes_names = X.columns.tolist()
            # replace nan values with None
            X = X.where(pd.notnull(X), None)
            self._X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            self._attributes_names = [
                f"{self.AUTOMATIC_ATTRIBUTES_NAMES_PREFIX}{index + 1}"
                for index in range(X.shape[1])
            ]
            self._X = X
        else:
            raise ValueError(
                f"Invalid X type: {str(type(X))}. "
                "Supported types are: 2 dimensional numpy array or pandas DataFrame "
                "object."
            )

    def _validate_X(self):
        if len(self._X.shape) != 2:
            raise ValueError(
                "X must be a 2 dimensional numpy array or pandas DataFrame object. "
                + f"Its current shape is: {str(self._X.shape)}"
            )

    def _validate_y(self):
        if self._y is not None and len(self._y.shape) != 1:
            raise ValueError(
                "y must be a 1 dimensional numpy array or pandas DataFrame object. "
                + f"Its current shape is: {str(self._y.shape)}"
            )

    def _create_example_set(self) -> JObject:
        data: JObject = self._prepare_data()
        args: list = [
            data,
            self._attributes_names,
            self._label_name,
            self._survival_time_attribute,
            self._contrast_attribute,
        ]
        DataTable = JClass(  # pylint: disable=invalid-name
            "adaa.analytics.rules.data.DataTable"
        )
        try:
            table = DataTable(*args)
            if self._y is not None:
                ExampleSetFactory = JClass(
                    'adaa.analytics.rules.logic.representation.'
                    'exampleset.ExampleSetFactory'
                )
                factory = ExampleSetFactory(2)
                example_set = factory.create(table)
                return example_set
            else:
                return table
        except Exception as error:
            from rulekit.exceptions import RuleKitJavaException

            RuleKitJavaException(error).print_java_stack_trace()
            raise error

    def _wrap_training_example_set(self, example_set: JObject) -> JObject:
        # training dataset must be wrapped in additional classes for rule induction
        # to work properly
        ExampleSetFactory = JClass(
            "adaa.analytics.rules.logic.representation.exampleset.ExampleSetFactory"
        )
        factory: JObject = ExampleSetFactory(self._problem_type.value)
        return factory.create(example_set)

    def _prepare_data(self) -> JObject:
        if self._y is None:
            data = self._X
        else:
            data = np.hstack((self._X.astype(object), self._y.reshape(-1, 1)))
        java_data = JObject(data, JArray("java.lang.Object", 2))
        return java_data


class PredictionResultMapper:
    """Maps prediction results to numpy array"""

    PREDICTION_COLUMN_ROLE: str = "prediction"
    CONFIDENCE_COLUMN_ROLE: str = "confidence"

    @staticmethod
    def map_confidence(predicted_example_set, label_unique_values: list) -> np.ndarray:
        """Maps models confidence values to numpy array

        Args:
            predicted_example_set (_type_): predicted ExampleSet instance
            label_unique_values (list): unique labels values

        Returns:
            np.ndarray: numpy array with mapped confidence values
        """
        confidence_matrix: list[list[float]] = []
        for label_value in label_unique_values:
            confidence_col: JObject = PredictionResultMapper._get_column_by_role(
                predicted_example_set,
                f"{PredictionResultMapper.CONFIDENCE_COLUMN_ROLE}_{label_value}",
            )

            confidence_values = [
                float(predicted_example_set.getExample(i).getValue(confidence_col))
                for i in range(predicted_example_set.size())
            ]

            confidence_matrix.append(confidence_values)
        return np.array(confidence_matrix, dtype=float).T

    @staticmethod
    def map(predicted_example_set: JObject) -> np.ndarray:
        """Maps models predictions to numpy array

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        prediction_col: JObject = PredictionResultMapper._get_column_by_role(
            predicted_example_set, PredictionResultMapper.PREDICTION_COLUMN_ROLE
        )
        if prediction_col.isNominal():
            return PredictionResultMapper.map_to_nominal(predicted_example_set)
        return PredictionResultMapper.map_to_numerical(predicted_example_set)

    @staticmethod
    def map_to_nominal(predicted_example_set: JObject) -> np.ndarray:
        """Maps models predictions to nominal numpy array of strings

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        prediction_col: JObject = PredictionResultMapper._get_column_by_role(
            predicted_example_set, PredictionResultMapper.PREDICTION_COLUMN_ROLE
        )

        return np.array(
            [
                str(predicted_example_set.getExample(i).getNominalValue(prediction_col))
                for i in range(predicted_example_set.size())
            ],
            dtype=str,
        )

    @staticmethod
    def map_to_numerical(
        predicted_example_set: JObject, remap: bool = True
    ) -> np.ndarray:
        """Maps models predictions to numerical numpy array

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        prediction_col: JObject = PredictionResultMapper._get_column_by_role(
            predicted_example_set, PredictionResultMapper.PREDICTION_COLUMN_ROLE
        )
        label_mapping = predicted_example_set.getAttributes().getLabel().getMapping()
        if remap:
            predictions: list = [
                label_mapping.mapIndex(
                    int(predicted_example_set.getExample(i).getValue(prediction_col))
                )
                for i in range(predicted_example_set.size())
            ]
            predictions = list(map(lambda x: float(str(x)), predictions))
            return np.array(predictions)
        return np.array(
            [
                float(predicted_example_set.getExample(i).getValue(prediction_col))
                for i in range(predicted_example_set.size())
            ]
        )

    @staticmethod
    def map_survival(predicted_example_set) -> np.ndarray:
        """Maps survival models predictions to numpy array. Used as alternative to
        `map` method used in survival analysis

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        estimators = []
        attribute = predicted_example_set.getAttributes().get("estimator")
        example_set_iterator = predicted_example_set.iterator()
        while example_set_iterator.hasNext():
            example = example_set_iterator.next()
            example_estimator = str(example.getValueAsString(attribute))
            example_estimator = example_estimator.split(" ")
            _, example_estimator[0] = example_estimator[0].split(":")
            times = [
                float(example_estimator[i])
                for i in range(len(example_estimator) - 1)
                if i % 2 == 0
            ]
            probabilities = [
                float(example_estimator[i])
                for i in range(len(example_estimator))
                if i % 2 != 0
            ]
            estimator = {"times": times, "probabilities": probabilities}
            estimators.append(estimator)
        return np.array(estimators)

    @staticmethod
    def _get_column_by_role(predicted_example_set: JObject, role: str) -> JObject:
        return predicted_example_set.getAttributes().getColumnByRole(role)


class ModelSerializer:
    """Class for serializing models"""

    @staticmethod
    def serialize(real_model: object) -> bytes:
        """Serialize Java ruleset object.

        Args:
            real_model (object): Java ruleset object
        """
        in_memory_file = io.BytesIO()
        JPickler(in_memory_file).dump(real_model)
        serialized_bytes = in_memory_file.getvalue()
        in_memory_file.close()
        return serialized_bytes

    @staticmethod
    def deserialize(serialized_bytes: bytes) -> object:
        """Deserialize Java ruleset object from bytes.

        Args:
            serialized_bytes (bytes): serialized bytes

        Returns:
            object: deserialized Java ruleset object
        """
        if not RuleKit.initialized:
            RuleKit.init()
        in_memory_file = io.BytesIO(serialized_bytes)
        model = JUnpickler(in_memory_file).load()
        in_memory_file.close()
        return model
