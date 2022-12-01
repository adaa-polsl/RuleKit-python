from typing import Union, Dict, List, Any
from jpype import JClass, JString, JObject, JArray, java
from jpype.pickle import JPickler, JUnpickler
import io
from warnings import warn
import numpy as np
import pandas as pd
from .params import *
from .rules import Rule
from rulekit import RuleKit


def get_rule_generator(expert: bool = False) -> Any:
    RuleKit.init()
    OperatorDocumentation = JClass(
        'com.rapidminer.tools.documentation.OperatorDocumentation')
    OperatorDescription = JClass('com.rapidminer.operator.OperatorDescription')
    Mockito = JClass('org.mockito.Mockito')
    path = 'adaa.analytics.rules.operator.'
    if expert:
        path += 'ExpertRuleGenerator'
    else:
        path += 'RuleGenerator'
    RuleGenerator = JClass(path)
    documentation = Mockito.mock(OperatorDocumentation.class_)
    description = Mockito.mock(OperatorDescription.class_)
    Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
    Mockito.when(description.getOperatorDocumentation()
                 ).thenReturn(documentation, None)
    return RuleGenerator(description)


class RuleGeneratorConfigurator:

    def __init__(self, rule_generator):
        self.rule_generator = rule_generator
        self.LogRank = None

    def configure(self, **kwargs: Dict[str, Any]) -> Any:
        self._configure_rule_generator(**kwargs)
        return self.rule_generator

    def configure_expert_parameter(self, param_name: str, param_value: Any):
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

    def configure_simple_parameter(self, param_name: str, param_value: Any):
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

    def _configure_rule_generator(self, **kwargs: Dict[str, Any]):
        if kwargs.get('min_rule_covered', None) is not None:
            # backward compatibility
            # TODO remove in version 2.0.0
            warn('"min_rule_covered" parameter was renamed to "minsupp_new" and is now deprecated, use "minsupp_new" instead. "min_rule_covered" parameter will be removed in next major version of the package. ', DeprecationWarning, stacklevel=6)
            kwargs['minsupp_new'] = kwargs['min_rule_covered']
        else:
            kwargs['min_rule_covered'] = kwargs['minsupp_new']

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
            self.configure_simple_parameter(param_name, param_value)


def map_attributes_names(example_set, attributes_names: List[str]):
    for index, name in enumerate(attributes_names):
        example_set.getAttributes().get(f'att{index + 1}').setName(name)


def set_attribute_role(example_set, attribute: str, role: str) -> object:
    OperatorDocumentation = JClass(
        'com.rapidminer.tools.documentation.OperatorDocumentation')
    OperatorDescription = JClass('com.rapidminer.operator.OperatorDescription')
    Mockito = JClass('org.mockito.Mockito')
    ChangeAttributeRole = JClass(
        'com.rapidminer.operator.preprocessing.filter.ChangeAttributeRole')

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


def _fix_missing_values(column) -> Any:
    for i in range(0, len(column.values)):
        if column.values[i] == b'?':
            column.values[i] = None


def create_example_set(
    values,
    labels=None,
    numeric_labels=False,
    survival_time_attribute: str = None,
    contrast_attribute: str = None,
) -> object:
    if labels is None:
        labels = ['' if not numeric_labels else 0] * len(values)
    attributes_names = None
    label_name = None
    if isinstance(values, pd.DataFrame):
        attributes_names = values.columns.values
        values = values.to_numpy()
    if isinstance(labels, pd.Series):
        label_name = labels.name
        labels = labels.to_numpy()
    values = JObject(values, JArray('java.lang.Object', 2))
    labels = JObject(labels, JArray('java.lang.Object', 1))
    ExampleSetFactory = JClass('com.rapidminer.example.ExampleSetFactory')
    example_set = ExampleSetFactory.createExampleSet(values, labels)
    if attributes_names is not None:
        map_attributes_names(example_set, attributes_names)
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

    @staticmethod
    def map_confidence(predicted_example_set, label_unique_values: list) -> np.ndarray:
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
        attribute = predicted_example_set.getAttributes().get('prediction')
        if attribute.isNominal():
            return PredictionResultMapper.map_to_nominal(predicted_example_set)
        else:
            return PredictionResultMapper.map_to_numerical(predicted_example_set)

    @staticmethod
    def map_to_nominal(predicted_example_set) -> np.ndarray:
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        label_mapping = attribute.getMapping()
        while row_reader.hasNext():
            row = row_reader.next()
            value_index = row.get(attribute)
            value = label_mapping.mapIndex(round(value_index))
            prediction.append(value)
        prediction = list(map(str, prediction))
        return np.array(prediction).astype(np.unicode_)

    @staticmethod
    def map_to_numerical(predicted_example_set, remap: bool = True) -> np.ndarray:
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
        estimators = []
        attribute = predicted_example_set.getAttributes().get("estimator")
        example_set_iterator = predicted_example_set.iterator()
        while example_set_iterator.hasNext():
            example = example_set_iterator.next()
            example_estimator = str(example.getValueAsString(attribute))
            example_estimator = example_estimator.split(" ")
            number_of_points, example_estimator[0] = example_estimator[0].split(
                ":")
            times = [float(example_estimator[i])
                     for i in range(len(example_estimator) - 1) if i % 2 == 0]
            probabilities = [float(example_estimator[i])
                             for i in range(len(example_estimator)) if i % 2 != 0]
            estimator = {'times': times, 'probabilities': probabilities}
            estimators.append(estimator)
        return np.array(estimators)


class ModelSerializer:

    @staticmethod
    def serialize(real_model: object) -> bytes:
        in_memory_file = io.BytesIO()
        JPickler(in_memory_file).dump(real_model)
        serialized_bytes = in_memory_file.getvalue()
        in_memory_file.close()
        return serialized_bytes

    @staticmethod
    def deserialize(serialized_bytes: bytes) -> object:
        in_memory_file = io.BytesIO(serialized_bytes)
        model = JUnpickler(in_memory_file).load()
        in_memory_file.close()
        return model
