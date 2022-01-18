from typing import Any, Union
from jpype import JClass, JObject, JArray, java
from .params import Measures
from .rules import Rule


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
