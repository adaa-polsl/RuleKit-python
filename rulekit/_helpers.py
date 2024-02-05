"""Contains helper functions and classes
"""
from typing import Union, Any
from warnings import warn
import io
from jpype import JClass, JString, JObject, JArray, java
from jpype.pickle import JPickler, JUnpickler
import numpy as np
import pandas as pd

from .params import (
    Measures,
    CONTRAST_ATTR_ROLE,
    SURVIVAL_TIME_ATTR_ROLE
)
from .rules import Rule
from .main import RuleKit


def get_rule_generator(expert: bool = False) -> Any:
    """Factory for Java RuleGenerator class object

    Args:
        expert (bool, optional): Whether expert induction is enables. Defaults to False.

    Returns:
        Any: RuleGenerator instance
    """
    RuleKit.init()
    OperatorDocumentation = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.tools.documentation.OperatorDocumentation')
    OperatorDescription = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.operator.OperatorDescription'
    )
    Mockito = JClass('org.mockito.Mockito')  # pylint: disable=invalid-name
    path: str = 'adaa.analytics.rules.operator.'
    if expert:
        path += 'ExpertRuleGenerator'
    else:
        path += 'RuleGenerator'
    RuleGenerator = JClass(path)  # pylint: disable=invalid-name
    documentation = Mockito.mock(OperatorDocumentation.class_)
    description = Mockito.mock(OperatorDescription.class_)
    Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
    Mockito.when(description.getOperatorDocumentation()
                 ).thenReturn(documentation, None)
    return RuleGenerator(description)


class RuleGeneratorConfigurator:
    """Class for configuring rule induction parameters
    """

    def __init__(self, rule_generator):
        self.rule_generator = rule_generator
        self.LogRank = None  # pylint: disable=invalid-name

    def configure(self, **kwargs: dict[str, Any]) -> Any:
        """Configures RuleGenerator instance with given induction parameters

        Returns:
            Any: configured RuleGenerator instance
        """
        self._configure_rule_generator(**kwargs)
        return self.rule_generator

    def _configure_expert_parameter(self, param_name: str, param_value: Any):
        if param_value is None:
            return
        rules_list = java.util.ArrayList()
        if isinstance(param_value, list) and len(param_value) > 0:
            if isinstance(param_value[0], str):
                for index, rule in enumerate(param_value):
                    rule_name = f'{param_name[:-1]}-{index}'
                    rules_list.add(
                        JObject([rule_name, rule], JArray('java.lang.String', 1)))
            elif isinstance(param_value[0], Rule):
                for index, rule in enumerate(param_value):
                    rule_name = f'{param_name[:-1]}-{index}'
                    rules_list.add(
                        JObject([rule_name, str(rule)], JArray('java.lang.String', 1)))
            elif isinstance(param_value[0], tuple):
                for index, rule in enumerate(param_value):
                    rules_list.add(
                        JObject([rule[0], rule[1]], JArray('java.lang.String', 1)))
        self.rule_generator.setListParameter(param_name, rules_list)

    def _configure_simple_parameter(self, param_name: str, param_value: Any):
        if param_value is not None:
            if isinstance(param_value, bool):
                param_value = (str(param_value)).lower()
            elif not isinstance(param_value, str):
                param_value = str(param_value)
            self.rule_generator.setParameter(param_name, param_value)

    def _configure_measure_parameter(self, param_name: str, param_value: Union[str, Measures]):
        if param_value is not None:
            if isinstance(param_value, Measures):
                self.rule_generator.setParameter(
                    param_name, param_value.value)
            if isinstance(param_value, str):
                self.rule_generator.setParameter(param_name, 'UserDefined')
                self.rule_generator.setParameter(param_name, param_value)

    def _configure_rule_generator(self, **kwargs: dict[str, Any]):
        if 'min_rule_covered' in kwargs and kwargs.get('min_rule_covered', None) is not None:
            # backward compatibility
            # TODO remove in version 2.0.0
            warn(
                '"min_rule_covered" parameter was renamed to "minsupp_new" and is now ' +
                'deprecated, "minsupp_new" instead. "min_rule_covered" parameter will be removed' +
                ' in next major version of the package. ',
                DeprecationWarning,
                stacklevel=6
            )
            kwargs['minsupp_new'] = kwargs['min_rule_covered']
            del kwargs['min_rule_covered']

        if kwargs.get('induction_measure') == Measures.LogRank or \
                kwargs.get('pruning_measure') == Measures.LogRank or \
                kwargs.get('voting_measure') == Measures.LogRank:
            self.LogRank = JClass('adaa.analytics.rules.logic.quality.LogRank')
        for measure_param_name in ['induction_measure', 'pruning_measure', 'voting_measure']:
            measure_param_value: Measures = kwargs.pop(
                measure_param_name, None)
            self._configure_measure_parameter(
                measure_param_name, measure_param_value)
        for param_name, param_value in kwargs.items():
            self._configure_simple_parameter(param_name, param_value)


def set_attributes_names(example_set, attributes_names: list[str]):
    """Sets attributes names

    Args:
        example_set (_type_): ExampleSet
        attributes_names (list[str]): attributes names
    """
    for index, name in enumerate(attributes_names):
        example_set.getAttributes().get(f'att{index + 1}').setName(name)


def set_attribute_role(example_set, attribute: str, role: str) -> object:
    """Set attribute special role

    Args:
        example_set (_type_): ExampleSet
        attribute (str): attribute name
        role (str): attribute role

    Returns:
        object: _description_
    """
    OperatorDocumentation = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.tools.documentation.OperatorDocumentation'
    )
    OperatorDescription = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.operator.OperatorDescription'
    )
    Mockito = JClass('org.mockito.Mockito')  # pylint: disable=invalid-name
    ChangeAttributeRole = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.operator.preprocessing.filter.ChangeAttributeRole'
    )

    documentation = Mockito.mock(OperatorDocumentation.class_)
    description = Mockito.mock(OperatorDescription.class_)
    Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
    Mockito.when(description.getOperatorDocumentation()
                 ).thenReturn(documentation, None)
    role_setter = ChangeAttributeRole(description)
    role_setter.setParameter(
        ChangeAttributeRole.PARAMETER_NAME, attribute)
    role_setter.setParameter(
        ChangeAttributeRole.PARAMETER_TARGET_ROLE, role)
    return role_setter.apply(example_set)


def _sanitize_dataset_columns(
    data: pd.DataFrame
) -> pd.DataFrame:
    for column_index in range(data.shape[1]):
        if data.iloc[:, column_index].dtypes.name == 'bool':
            # ExampleSet class that RuleKit internally uses does not
            # support boolean columns at the moment (see Issue #18)
            data.iloc[:, column_index] = data.iloc[:, column_index].astype(str)
    return data


def create_example_set(
    values: Union[pd.DataFrame, np.ndarray],
    labels: Union[pd.Series, np.ndarray] = None,
    numeric_labels: bool = False,
    survival_time_attribute: str = None,
    contrast_attribute: str = None,
) -> object:
    """Creates Java RapidMiner ExampleSet object instance

    Args:
        values (Union[pd.DataFrame, np.ndarray]): Attributes values
        labels (Union[pd.Series, np.ndarray], optional): Labels. Defaults to None.
        numeric_labels (bool, optional): Whether labels are numerical or not . Defaults to False.
        survival_time_attribute (str, optional): Name of special survival time attribute. Used 
            for survival analysis Defaults to None.
        contrast_attribute (str, optional): Name of special contrast attribute. Used 
            for contrast sets analysis Defaults to None.

    Returns:
        object: ExampleSet object instance
    """
    if labels is None:
        labels = ['' if not numeric_labels else 0] * len(values)
    attributes_names = None
    label_name = None
    if isinstance(values, pd.DataFrame):
        values = _sanitize_dataset_columns(values)
        attributes_names = values.columns.values
        values = values.to_numpy()
    if isinstance(labels, pd.Series):
        label_name = labels.name
        labels = labels.to_numpy()
    values = JObject(values, JArray('java.lang.Object', 2))
    labels = JObject(labels, JArray('java.lang.Object', 1))
    ExampleSetFactory = JClass(  # pylint: disable=invalid-name
        'com.rapidminer.example.ExampleSetFactory'
    )
    example_set = ExampleSetFactory.createExampleSet(values, labels)
    if attributes_names is not None:
        set_attributes_names(example_set, attributes_names)
    if label_name is not None:
        example_set.getAttributes().get('label').setName(label_name)
    if survival_time_attribute is not None:
        if survival_time_attribute == '':
            survival_time_attribute = f'att{example_set.getAttributes().size()}'
        example_set = set_attribute_role(
            example_set, survival_time_attribute, SURVIVAL_TIME_ATTR_ROLE)
    if contrast_attribute is not None:
        example_set = set_attribute_role(
            example_set, contrast_attribute, CONTRAST_ATTR_ROLE)
    return example_set


class PredictionResultMapper:
    """Maps prediction results to numpy array
    """

    @staticmethod
    def map_confidence(
        predicted_example_set,
        label_unique_values: list
    ) -> np.ndarray:
        """Maps models confidence values to numpy array

        Args:
            predicted_example_set (_type_): predicted ExampleSet instance
            label_unique_values (list): unique labels values

        Returns:
            np.ndarray: numpy array with mapped confidence values
        """
        confidence_attributes_names = list(
            map(lambda val: f'confidence_{val}', label_unique_values))
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        confidence_attributes = []
        for name in confidence_attributes_names:
            confidence_attributes.append(
                predicted_example_set.getAttributes().get(name))
        while row_reader.hasNext():
            row = row_reader.next()
            value = []
            for attribute in confidence_attributes:
                value.append(attribute.getValue(row))
            prediction.append(np.array(value))
        return np.array(prediction)

    @staticmethod
    def map(predicted_example_set) -> np.ndarray:
        """Maps models predictions to numpy array

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        attribute = predicted_example_set.getAttributes().get('prediction')
        if attribute.isNominal():
            return PredictionResultMapper.map_to_nominal(predicted_example_set)
        else:
            return PredictionResultMapper.map_to_numerical(predicted_example_set)

    @staticmethod
    def map_to_nominal(predicted_example_set) -> np.ndarray:
        """Maps models predictions to nominal numpy array of strings

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        prediction: list[str] = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        label_mapping = attribute.getMapping()
        while row_reader.hasNext():
            row = row_reader.next()
            value_index = row.get(attribute)
            value = label_mapping.mapIndex(round(value_index))
            prediction.append(value)
        prediction = [str(e) for e in prediction]
        return np.array(prediction).astype(np.unicode_)

    @staticmethod
    def map_to_numerical(predicted_example_set, remap: bool = True) -> np.ndarray:
        """Maps models predictions to numerical numpy array

        Args:
            predicted_example_set (_type_): ExampleSet with predictions

        Returns:
            np.ndarray: numpy array containing predictions
        """
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        label_mapping = predicted_example_set.getAttributes().getLabel().getMapping()
        while row_reader.hasNext():
            row = row_reader.next()
            if remap:
                value = int(attribute.getValue(row))
                value = float(str(label_mapping.mapIndex(value)))
            else:
                value = float(attribute.getValue(row))
            prediction.append(value)
        return np.array(prediction)

    @staticmethod
    def map_survival(predicted_example_set) -> np.ndarray:
        """Maps survival models predictions to numpy array. Used as alternative to `map` method 
        used in survival analysis

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
            _, example_estimator[0] = example_estimator[0].split(
                ":")
            times = [
                float(example_estimator[i])
                for i in range(len(example_estimator) - 1) if i % 2 == 0
            ]
            probabilities = [float(example_estimator[i])
                             for i in range(len(example_estimator)) if i % 2 != 0]
            estimator = {'times': times, 'probabilities': probabilities}
            estimators.append(estimator)
        return np.array(estimators)


class ModelSerializer:
    """Class for serializing models
    """

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
        in_memory_file = io.BytesIO(serialized_bytes)
        model = JUnpickler(in_memory_file).load()
        in_memory_file.close()
        return model
