from typing import Iterable
from numbers import Number
from jpype import JClass, JObject, JString, JArray


class DecisionTreeClassifier:

    def __init__(self):
        OperatorDocumentation = JClass('com.rapidminer.tools.documentation.OperatorDocumentation')
        OperatorDescription = JClass('com.rapidminer.operator.OperatorDescription')
        Mockito = JClass('org.mockito.Mockito')
        RuleGenerator = JClass('adaa.analytics.rules.operator.RuleGenerator')

        documentation = Mockito.mock(OperatorDocumentation.class_)
        description = Mockito.mock(OperatorDescription.class_)
        Mockito.when(documentation.getShortName()).thenReturn(JString(''), None)
        Mockito.when(description.getOperatorDocumentation()).thenReturn(documentation, None)
        self.rule_generator = RuleGenerator(description)
        self.model = None

    def _map_result(self, predicted_example_set) -> Iterable:
        prediction = []
        row_reader = predicted_example_set.getExampleTable().getDataRowReader()
        attribute = predicted_example_set.getAttributes().get('prediction')
        while row_reader.hasNext():
            row = row_reader.next()
            value_index = row.get(attribute)
            value = attribute.getMapping().mapIndex(round(value_index))
            prediction.append(value)
        return prediction

    def fit(self, values: Iterable[Iterable], labels: Iterable) -> object:
        if isinstance(values[0][0], Number) and isinstance(labels[0], Number):
            values = JObject(values, JArray('java.lang.Double', 2))
        else:
            values = JObject(values, JArray('java.lang.Object', 2))
        ExampleSetFactory = JClass('com.rapidminer.example.ExampleSetFactory')
        example_set = ExampleSetFactory.createExampleSet(values, labels)
        self.model = self.rule_generator.learn(example_set)
        return self

    def predict(self, values: Iterable) -> Iterable:
        labels = [values[0][0]] * len(values)
        values = JObject(values, JArray('java.lang.Object', 2))
        labels = JObject(labels, JArray('java.lang.Object', 1))
        ExampleSetFactory = JClass('com.rapidminer.example.ExampleSetFactory')
        testExampleSet = ExampleSetFactory.createExampleSet(values, labels)
        result = self.model.apply(testExampleSet)
        return self._map_result(result)