from jpype import JClass, JString, JObject, JArray
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
        voting_measure: Measures):
    if min_rule_covered is not None:
        rule_generator.setParameter('min_rule_covered', str(min_rule_covered))
    if induction_measure is not None:
        if isinstance(induction_measure, Measures):
            rule_generator.setParameter('induction_measure', induction_measure.value)
        if isinstance(induction_measure, str):
            rule_generator.setParameter('induction_measure', 'UserDefined')
            rule_generator.setParameter('induction_measure', induction_measure)
    if pruning_measure is not None:
        if isinstance(pruning_measure, Measures):
            rule_generator.setParameter('pruning_measure', pruning_measure.value)
        if isinstance(pruning_measure, str):
            rule_generator.setParameter('pruning_measure', 'UserDefined')
            rule_generator.setParameter('user_pruning_equation', pruning_measure)
    if voting_measure is not None:
        if isinstance(voting_measure, Measures):
            rule_generator.setParameter('voting_measure', voting_measure.value)
        if isinstance(voting_measure, str):
            rule_generator.setParameter('voting_measure', 'UserDefined')
            rule_generator.setParameter('voting_measure', voting_measure)


def create_example_set(values, labels=None, numeric_labels=False) -> object:
    if labels is None:
        labels = ['' if not numeric_labels else 0] * len(values)
    if isinstance(values, pd.DataFrame):
        values = values.to_numpy()
    if isinstance(labels, pd.DataFrame):
        labels = labels.to_numpy()
    values = JObject(values, JArray('java.lang.Object', 2))
    labels = JObject(labels, JArray('java.lang.Object', 1))
    ExampleSetFactory = JClass('com.rapidminer.example.ExampleSetFactory')
    return ExampleSetFactory.createExampleSet(values, labels)


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
