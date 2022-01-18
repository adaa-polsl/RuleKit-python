from jpype import JClass, JString
from jpype.types import JDouble

from ..conditions import CompoundCondition, ElementaryCondition
from ..rules import Rule


class ClassificationRuleBuilder:

    _java_class_path = 'adaa.analytics.rules.utils.builder.ClassificationRuleBuilder'

    def __init__(self, label_attribute_name: str = 'label'):
        java_class = JClass(ClassificationRuleBuilder._java_class_path)
        self._java_object = java_class(JString(label_attribute_name))

    @staticmethod
    def from_rule(rule: Rule, label_attribute_name: str) -> object:
        instance = ClassificationRuleBuilder(label_attribute_name)
        if rule._java_object.getClass() != JClass('adaa.analytics.rules.logic.representation.ClassificationRule'):
            raise ValueError('Rule must be classification rule')
        ClassificationRuleBuilderJavaClass = JClass(
            ClassificationRuleBuilder._java_class_path)
        instance._java_object = ClassificationRuleBuilderJavaClass(
            rule._java_object, JString(label_attribute_name))
        return instance

    def build(self) -> Rule:
        java_rule = self._java_object.build()
        return Rule(java_rule)

    def setConsequence(self, label_value: float):
        self._java_object.setConsequence(JDouble(label_value))
        return self

    def setPremise(self, premise: ElementaryCondition):
        self._java_object.setPremise(premise._java_object)
        return self

    def setPremise(self, premise: CompoundCondition):
        self._java_object.setPremise(premise._java_object)
        return self

