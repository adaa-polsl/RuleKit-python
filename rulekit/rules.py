from typing import Union, List
from .params import Measures
import numpy as np
from .stats import RuleStatistics, RuleSetStatistics


class Rule:

    def __init__(self, java_object):
        self._java_object = java_object
        self._stats: RuleStatistics = None

    @property
    def weight(self) -> float:
        return self._java_object.getWeight()

    @property
    def weighted_p(self) -> float:
        return self._java_object.getWeighted_p()

    @property
    def weighted_n(self) -> float:
        return self._java_object.getWeighted_n()

    @property
    def weighted_P(self) -> float:
        return self._java_object.getWeighted_P()

    @property
    def weighted_N(self) -> float:
        return self._java_object.getWeighted_N()

    @property
    def pvalue(self) -> float:
        return self._java_object.getPValue()

    @property
    def stats(self) -> RuleStatistics:
        if self._stats is None:
            self._stats = RuleStatistics(self)
        return self._stats

    def get_covering_information(self) -> dict:
        return {
            'weighted_n': self.weighted_n,
            'weighted_p': self.weighted_p,
            'weighted_N': self.weighted_N,
            'weighted_P': self.weighted_P,
        }

    def print_stats(self):
        print(self.stats)

    def __str__(self):
        return str(self._java_object.toString())


class InductionParameters:

    def __init__(self, java_object):
        self._java_object = java_object

        self.minimum_covered: float = self._java_object.getMinimumCovered()
        self.maximum_uncovered_fraction: float = self._java_object.getMaximumUncoveredFraction()
        self.ignore_missing: bool = self._java_object.isIgnoreMissing()
        self.pruning_enabled: bool = self._java_object.isPruningEnabled()
        self.max_growing_condition: float = self._java_object.getMaxGrowingConditions()

    @property
    def induction_measure(self) -> Union[Measures, str]:
        return InductionParameters._get_measure_str(self._java_object.getInductionMeasure())

    @property
    def pruning_measure(self) -> Union[Measures, str]:
        return InductionParameters._get_measure_str(self._java_object.getPruningMeasure())

    @property
    def voting_measure(self) -> Union[Measures, str]:
        return InductionParameters._get_measure_str(self._java_object.getVotingMeasure())

    @staticmethod
    def _get_measure_str(measure) -> Union[Measures, str]:
        name: str = measure.getName()
        if name == 'UserDefined':
            return 'UserDefined'
        else:
            return Measures[name]

    def __str__(self):
        return str(self._java_object.toString())


class RuleSet:

    def __init__(self, java_object):
        self._java_object = java_object
        self._stats: RuleSetStatistics = None

    @property
    def total_time(self) -> float:
        return self._java_object.getTotalTime()

    @property
    def growing_time(self) -> float:
        return self._java_object.getGrowingTime()

    @property
    def pruning_time(self) -> float:
        return self._java_object.getPruningTime()

    @property
    def is_voting(self) -> bool:
        return self._java_object.getIsVoting()

    @property
    def parameters(self) -> object:
        return InductionParameters(self._java_object.getParams())

    @property
    def stats(self) -> RuleSetStatistics:
        if self._stats is None:
            self._stats = RuleSetStatistics(self)
        return self._stats

    def covers(self, example_set) -> list:
        res = []
        for rule in self.rules:
            covering_info = rule._java_object.covers(example_set)
            covered_examples_indexes = None
            if len(covering_info.positives) > 0:
                covered_examples_indexes = covering_info.positives
            elif len(covering_info.negatives) > 0:
                covered_examples_indexes = covering_info.negatives
            res.append(covered_examples_indexes)
        return np.array(res)

    @property
    def rules(self) -> List[Rule]:
        rules = self._java_object.getRules()
        return list(map(lambda rule: Rule(rule), rules))

    def calculate_conditions_count(self) -> float:
        return self._java_object.calculateConditionsCount()

    def calculate_induced_conditions_count(self) -> float:
        return self._java_object.calculateInducedCondtionsCount()

    def calculate_avg_rule_coverage(self) -> float:
        return self._java_object.calculateAvgRuleCoverage()

    def calculate_avg_rule_precision(self) -> float:
        return self._java_object.calculateAvgRulePrecision()

    def calculate_avg_rule_quality(self) -> float:
        return self._java_object.calculateAvgRuleQuality()

    def calculate_significance(self, alpha: float) -> dict:
        significance = self._java_object.calculateSignificance(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def calculate_significance_fdr(self, alpha: float) -> dict:
        significance = self._java_object.calculateSignificanceFDR(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def calculate_significance_fwer(self, alpha: float) -> dict:
        significance = self._java_object.calculateSignificanceFWER(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def __str__(self):
        return str(self._java_object.toString())
