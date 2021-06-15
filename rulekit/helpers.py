from typing import Union, List, Any
from jpype import JClass, JString, JObject, JArray, java, addClassPath
from jpype.pickle import JPickler, JUnpickler
import io
import numpy as np
import pandas as pd
from .params import Measures
from .rules import Rule


def get_rule_generator(expert: bool = False) -> Any:
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

    def configure(self,
                  min_rule_covered: int = None,
                  induction_measure: Measures = None,
                  pruning_measure: Union[Measures, str] = None,
                  voting_measure: Measures = None,
                  max_growing: int = None,
                  enable_pruning: bool = None,
                  ignore_missing: bool = None,
                  max_uncovered_fraction: float = None,
                  select_best_candidate: bool = None,
                  survival_time_attr: str = None,

                  extend_using_preferred: bool = None,
                  extend_using_automatic: bool = None,
                  induce_using_preferred: bool = None,
                  induce_using_automatic: bool = None,
                  consider_other_classes: bool = None,
                  preferred_attributes_per_rule: int = None,
                  preferred_conditions_per_rule: int = None) -> Any:
        self._configure_rule_generator(
            min_rule_covered=min_rule_covered,
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
            consider_other_classes=consider_other_classes,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule
        )
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

    def _configure_rule_generator(
            self,
            min_rule_covered: int = None,
            induction_measure: Measures = None,
            pruning_measure: Measures = None,
            voting_measure: Measures = None,
            max_growing: int = None,
            enable_pruning: bool = None,
            ignore_missing: bool = None,
            max_uncovered_fraction: float = None,
            select_best_candidate: bool = None,

            extend_using_preferred: bool = None,
            extend_using_automatic: bool = None,
            induce_using_preferred: bool = None,
            induce_using_automatic: bool = None,
            consider_other_classes: bool = None,
            preferred_conditions_per_rule: int = None,
            preferred_attributes_per_rule: int = None):
        if induction_measure == Measures.LogRank or pruning_measure == Measures.LogRank or voting_measure == Measures.LogRank:
            self.LogRank = JClass('adaa.analytics.rules.logic.quality.LogRank')
        self.configure_simple_parameter('min_rule_covered', min_rule_covered)
        self.configure_simple_parameter('max_growing', max_growing)
        self.configure_simple_parameter('enable_pruning', enable_pruning)
        self.configure_simple_parameter(
            'max_uncovered_fraction', max_uncovered_fraction)
        self.configure_simple_parameter(
            'select_best_candidate', select_best_candidate)

        self.configure_simple_parameter(
            'extend_using_preferred', extend_using_preferred)
        self.configure_simple_parameter(
            'extend_using_automatic', extend_using_automatic)
        self.configure_simple_parameter(
            'induce_using_preferred', induce_using_preferred)
        self.configure_simple_parameter(
            'induce_using_automatic', induce_using_automatic)
        self.configure_simple_parameter(
            'consider_other_classes', consider_other_classes)
        self.configure_simple_parameter(
            'preferred_conditions_per_rule', preferred_conditions_per_rule)
        self.configure_simple_parameter(
            'preferred_attributes_per_rule', preferred_attributes_per_rule)

        self._configure_measure_parameter(
            'induction_measure', induction_measure)
        self._configure_measure_parameter('pruning_measure', pruning_measure)
        self._configure_measure_parameter('voting_measure', voting_measure)


def map_attributes_names(example_set, attributes_names: List[str]):
    for index, name in enumerate(attributes_names):
        example_set.getAttributes().get(f'att{index + 1}').setName(name)


def set_survival_time(example_set, survival_time_attribute: str) -> object:
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
        ChangeAttributeRole.PARAMETER_NAME, survival_time_attribute)
    role_setter.setParameter(
        ChangeAttributeRole.PARAMETER_TARGET_ROLE, "survival_time")
    return role_setter.apply(example_set)


def _fix_missing_values(column) -> Any:
    for i in range(0, len(column.values)):
        if column.values[i] == b'?':
            column.values[i] = None


def create_example_set(values, labels=None, numeric_labels=False, survival_time_attribute: str = None) -> object:
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
        example_set = set_survival_time(example_set, survival_time_attribute)
    return example_set


def create_sorted_example_set(values, labels=None, numeric_labels=False, survival_time_attribute: str = None) -> object:
    example_set = create_example_set(
        values, labels, numeric_labels, survival_time_attribute)
    SortedExampleSet = JClass("com.rapidminer.example.set.SortedExampleSet")
    sorted_example_set = SortedExampleSet(
        example_set, example_set.getAttributes().getLabel(), SortedExampleSet.INCREASING
    )
    return sorted_example_set


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
