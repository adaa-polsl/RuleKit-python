from typing import List
import pandas as pd
import numpy as np
from jpype import JClass
from pandas.core.arrays import boolean


class BaseCondition:

    def __init__(self, java_object):
        self._java_object = java_object

    @property
    def prunable(self) -> boolean:
        return self._java_object.isPrunable()

    @property
    def attributes(self) -> np.ndarray:
        indices_set = self._java_object.getAttributes()
        indices_java_array = indices_set.toArray()
        return np.array(list(map(lambda e: str(e), indices_java_array)))

    def evaluate(self, example: pd.Series) -> boolean:
        raise Exception('Not implemented')

    def evaluate_on_dataset(self, dataset: pd.DataFrame):
        raise Exception('Not implemented')

    def __eq__(self, other): 
        return str(self) == str(other)

    def __ne__(self, other): 
        return not str(self) == str(other)

    def __str__(self) -> str:
        return str(self._java_object.toString())


class ElementaryCondition(BaseCondition):

    _java_class_path = 'adaa.analytics.rules.logic.representation.ElementaryCondition'

    @property
    def attribute(self) -> str:
        return str(self.attributes[0])

    def __init__(self, java_object):
        self._java_object = java_object


class CompoundCondition(BaseCondition):

    _java_class_path = 'adaa.analytics.rules.logic.representation.CompoundCondition'


    def __init__(self, java_object):
        self._java_object = java_object
        self._ElementaryConditionJavaClass = JClass(ElementaryCondition._java_class_path)
        self._CompoundConditionJavaClass = JClass(CompoundCondition._java_class_path)

    @property
    def subconditions(self) -> List[BaseCondition]:
        tmp = self._java_object.getSubconditions()
        subconditions = []
        for subcondition in tmp:
            if subcondition.getClass() == self._ElementaryConditionJavaClass:
                subconditions.append(ElementaryCondition(subcondition))
            if subcondition.getClass() == self._CompoundConditionJavaClass:
                subconditions.append(CompoundCondition(subcondition))
        return subconditions
