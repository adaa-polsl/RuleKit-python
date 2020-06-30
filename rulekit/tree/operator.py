from .helpers import get_rule_generator, configure_rule_generator, create_example_set, PredictionResultMapper
from .params import Measures
from .rules import RuleSet
import numpy as np
from typing import Union, Iterable


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
        configure_rule_generator(
            self._rule_generator,
            min_rule_covered,
            induction_measure,
            pruning_measure,
            voting_measure,
            max_growing,
            enable_pruning,
            ignore_missing
        )
        self.model: RuleSet = None
        self._real_model = None

    def fit(self, values: Iterable[Iterable], labels: Iterable, survival_time_attribute: str = None) -> object:
        example_set = create_example_set(values, labels, survival_time_attribute=survival_time_attribute)
        self._real_model = self._rule_generator.learn(example_set)
        self.model = RuleSet(self._real_model)
        return self

    def predict(self, values: Iterable) -> np.ndarray:
        example_set = create_example_set(values)
        return self._real_model.apply(example_set)