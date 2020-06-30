from typing import Iterable, List
from jpype import JClass, JString, JObject, JArray, java
import numpy as np
import pandas as pd
from .params import Measures


def get_rule_generator() -> object:
    OperatorDocumentation = JClass('com.rapidminer.tools.documentation.OperatorDocumentation')
    OperatorDescription = JClass('com.rapidminer.operator.OperatorDescription')
    Mockito = JClass('org.mockito.Mockito')
    RuleGenerator = JClass('adaa.analytics.rules.operator.RuleGenerator')

    documentation = Mockito.mock(OperatorDocumentation.class_)
    description = Mockito.mock(OperatorDescription.class_)
    Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
    Mockito.when(description.getOperatorDocumentation()).thenReturn(documentation, None)
    return RuleGenerator(description)


def configure_rule_generator(
        rule_generator,
        min_rule_covered: int,
        induction_measure: Measures,
        pruning_measure: Measures,
        voting_measure: Measures,
        max_growing: int = 0,
        enable_pruning: bool = True,
        ignore_missing: bool = False):
    # TODO przerobić te metode
    if induction_measure == Measures.LogRank or pruning_measure == Measures.LogRank or voting_measure == Measures.LogRank:
        LogRank = JClass('adaa.analytics.rules.logic.quality.LogRank')
    if min_rule_covered is not None:
        rule_generator.setParameter('min_rule_covered', str(min_rule_covered))
    if induction_measure is not None:
        if isinstance(induction_measure, Measures):
            if induction_measure == Measures.LogRank:
                rule_generator.setInductionMeasure(LogRank())
            else:
                rule_generator.setParameter('induction_measure', induction_measure.value)
        if isinstance(induction_measure, str):
            rule_generator.setParameter('induction_measure', 'UserDefined')
            rule_generator.setParameter('induction_measure', induction_measure)
    if pruning_measure is not None:
        if isinstance(pruning_measure, Measures):
            if pruning_measure == Measures.LogRank:
                rule_generator.setPruningMeasure(LogRank())
            else:
                rule_generator.setParameter('pruning_measure', pruning_measure.value)
        if isinstance(pruning_measure, str):
            rule_generator.setParameter('pruning_measure', 'UserDefined')
            rule_generator.setParameter('user_pruning_equation', pruning_measure)
    if voting_measure is not None:
        if isinstance(voting_measure, Measures):
            if voting_measure == Measures.LogRank:
                rule_generator.setVotingMeasure(LogRank())
            else:
                rule_generator.setParameter('voting_measure', voting_measure.value)
        if isinstance(voting_measure, str):
            rule_generator.setParameter('voting_measure', 'UserDefined')
            rule_generator.setParameter('voting_measure', voting_measure)
    if max_growing is not None:
        rule_generator.setParameter('max_growing', max_growing)
    if enable_pruning is not None:
        rule_generator.setParameter('enable_pruning', enable_pruning)
    if ignore_missing is not None:
        rule_generator.setParameter('ignore_missing', ignore_missing)


def map_attributes_names(example_set, attributes_names: List[str]):
    for index, name in enumerate(attributes_names):
        example_set.getAttributes().get(f'att{index + 1}').setName(name)


def set_survival_time(example_set, survival_time_attribute: str) -> object:
    OperatorDocumentation = JClass('com.rapidminer.tools.documentation.OperatorDocumentation')
    OperatorDescription = JClass('com.rapidminer.operator.OperatorDescription')
    Mockito = JClass('org.mockito.Mockito')
    ChangeAttributeRole = JClass('com.rapidminer.operator.preprocessing.filter.ChangeAttributeRole')

    documentation = Mockito.mock(OperatorDocumentation.class_)
    description = Mockito.mock(OperatorDescription.class_)
    Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
    # Mockito.when(documentation.getHTMLMessage()).thenReturn(JString(''), None)
    Mockito.when(description.getOperatorDocumentation()).thenReturn(documentation, None)
    role_setter = ChangeAttributeRole(description)
    role_setter.setParameter(ChangeAttributeRole.PARAMETER_NAME, survival_time_attribute);
    role_setter.setParameter(ChangeAttributeRole.PARAMETER_TARGET_ROLE, "survival_time");
    return role_setter.apply(example_set)


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


class PredictionResultMapper:

    @staticmethod
    def map_to_nominal(predicted_example_set) -> np.ndarray:
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        while row_reader.hasNext():
            row = row_reader.next()
            value_index = row.get(attribute)
            value = attribute.getMapping().mapIndex(round(value_index))
            prediction.append(value)
        prediction = list(map(str, prediction))
        return np.array(prediction)

    @staticmethod
    def map_to_numerical(predicted_example_set) -> np.ndarray:
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        while row_reader.hasNext():
            row = row_reader.next()
            value = attribute.getValue(row)
            prediction.append(value)
        prediction = list(map(float, prediction))
        return np.array(prediction)
