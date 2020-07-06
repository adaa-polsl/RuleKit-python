from .helpers import RuleGeneratorConfigurator, PredictionResultMapper, create_example_set, get_rule_generator
from .params import Measures
from .rules import RuleSet, Rule
import numpy as np
from typing import Union, Iterable, Any, List

from jpype import JClass

class BaseOperator:

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None):
        self._rule_generator = get_rule_generator()
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = self._configurator.configure(
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing
        )
        self.model: RuleSet = None
        self._real_model = None

    def _map_result(self, predicted_example_set) -> np.ndarray:
        return PredictionResultMapper.map(predicted_example_set)

    def fit(self, values: Iterable[Iterable], labels: Iterable, survival_time_attribute: str = None) -> Any:
        example_set = create_example_set(values, labels, survival_time_attribute=survival_time_attribute)
        self._real_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(self._real_model)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        example_set = create_example_set(values)
        return self._real_model.apply(example_set)


class ExpertKnowledgeOperator:

    def __init__(self,
                 min_rule_covered: int = None,
                 induction_measure: Measures = None,
                 pruning_measure: Union[Measures, str] = None,
                 voting_measure: Measures = None,
                 max_growing: int = None,
                 enable_pruning: bool = None,
                 ignore_missing: bool = None,

                 extend_using_preferred: bool = None,
                 extend_using_automatic: bool = None,
                 induce_using_preferred: bool = None,
                 induce_using_automatic: bool = None,
                 consider_other_classes: bool = None,
                 preferred_conditions_per_rule: int = None,
                 preferred_attributes_per_rule: int = None):
        self._rule_generator = get_rule_generator(expert=True)
        self._configurator = RuleGeneratorConfigurator(self._rule_generator)
        self._rule_generator = self._configurator.configure(
            min_rule_covered=min_rule_covered,
            induction_measure=induction_measure,
            pruning_measure=pruning_measure,
            voting_measure=voting_measure,
            max_growing=max_growing,
            enable_pruning=enable_pruning,
            ignore_missing=ignore_missing,
            extend_using_preferred=extend_using_preferred,
            extend_using_automatic=extend_using_automatic,
            induce_using_preferred=induce_using_preferred,
            induce_using_automatic=induce_using_automatic,
            consider_other_classes=consider_other_classes,
            preferred_conditions_per_rule=preferred_conditions_per_rule,
            preferred_attributes_per_rule=preferred_attributes_per_rule
        )
        self.model: RuleSet = None
        self._real_model = None

    def fit(self,
            values: Iterable[Iterable],
            labels: Iterable,
            survival_time_attribute: str = None,

            expert_rules: List[Union[str, Rule]] = None,
            expert_preferred_conditions: List[Union[str, Rule]] = None,
            expert_forbidden_conditions: List[Union[str, Rule]] = None) -> Any:
        self._configurator._configure_simple_parameter('use_expert', True)
        self._configurator.configure_expert_parameter('expert_preferred_conditions', expert_preferred_conditions)
        self._configurator.configure_expert_parameter('expert_forbidden_conditions', expert_forbidden_conditions)
        self._configurator.configure_expert_parameter('expert_rules', expert_rules)
        example_set = create_example_set(values, labels, survival_time_attribute=survival_time_attribute)

        # FileOutputStream = JClass('java.io.FileOutputStream')
        # ObjectOutputStream = JClass('java.io.ObjectOutputStream')
        # fout = FileOutputStream("./example_set")
        # oos = ObjectOutputStream(fout)
        # oos.writeObject(example_set)

        self._real_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(self._real_model)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        example_set = create_example_set(values)
        return self._real_model.apply(example_set)
