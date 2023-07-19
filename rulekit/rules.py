"""Contains classes representing rules and rulesets.
"""
from typing import Union
import numpy as np
from .params import Measures
from .stats import RuleStatistics, RuleSetStatistics


class Rule:
    """Class representing single rule."""

    def __init__(self, java_object):
        """:meta private:"""
        self._java_object = java_object
        self._stats: RuleStatistics = None

    @property
    def weight(self) -> float:
        """Rule weight"""
        return self._java_object.getWeight()

    @property
    def weighted_p(self) -> float:
        """Number of positives covered by the rule (accounting weights)."""
        return self._java_object.getWeighted_p()

    @property
    def weighted_n(self) -> float:
        """Number of negatives covered by the rule (accounting weights)."""
        return self._java_object.getWeighted_n()

    @property
    def weighted_P(self) -> float:  # pylint: disable=invalid-name
        """Number of positives in the training set (accounting weights)."""
        return self._java_object.getWeighted_P()

    @property
    def weighted_N(self) -> float:  # pylint: disable=invalid-name
        """Number of negatives in the training set (accounting weights)."""
        return self._java_object.getWeighted_N()

    @property
    def pvalue(self) -> float:
        """Rule significance."""
        return self._java_object.getPValue()

    @property
    def stats(self) -> RuleStatistics:
        """Rule statistics."""
        if self._stats is None:
            self._stats = RuleStatistics(self)
        return self._stats

    def get_covering_information(self) -> dict:
        """Returns information about rule covering

        Returns
        -------
        covering_data : dict
            Dictionary containing covering information.
        """
        return {
            'weighted_n': self.weighted_n,
            'weighted_p': self.weighted_p,
            'weighted_N': self.weighted_N,
            'weighted_P': self.weighted_P,
        }

    def print_stats(self):
        """Prints rule statistics as formatted text."""
        print(self.stats)

    def __str__(self):
        """Returns string representation of the rule."""
        return str(self._java_object.toString())


class InductionParameters:
    """Induction parameters.
    """

    def __init__(self, java_object):
        self._java_object = java_object

        self.minimum_covered: float = self._java_object.getMinimumCovered()
        self.maximum_uncovered_fraction: float = self._java_object.getMaximumUncoveredFraction()
        self.ignore_missing: bool = self._java_object.isIgnoreMissing()
        self.pruning_enabled: bool = self._java_object.isPruningEnabled()
        self.max_growing_condition: float = self._java_object.getMaxGrowingConditions()

    @property
    def induction_measure(self) -> Union[Measures, str]:
        """
        Returns:
            Union[Measures, str]: Measure used for induction
        """
        return InductionParameters._get_measure_str(self._java_object.getInductionMeasure())

    @property
    def pruning_measure(self) -> Union[Measures, str]:
        """
        Returns:
            Union[Measures, str]: Measure used for pruning
        """
        return InductionParameters._get_measure_str(self._java_object.getPruningMeasure())

    @property
    def voting_measure(self) -> Union[Measures, str]:
        """
        Returns:
            Union[Measures, str]: Measure used for voting
        """
        return InductionParameters._get_measure_str(self._java_object.getVotingMeasure())

    @staticmethod
    def _get_measure_str(measure) -> Union[Measures, str]:
        name: str = measure.getName()
        if name == 'UserDefined':
            return 'UserDefined'
        return Measures[name]

    def __str__(self):
        return str(self._java_object.toString())


class RuleSet:
    """Class representing ruleset."""

    def __init__(self, java_object):
        """:meta private:"""
        self._java_object = java_object
        self._stats: RuleSetStatistics = None

    @property
    def total_time(self) -> float:
        """Time of constructing the rule set in seconds"""
        return self._java_object.getTotalTime()

    @property
    def growing_time(self) -> float:
        """Time of growing in seconds"""
        return self._java_object.getGrowingTime()

    @property
    def pruning_time(self) -> float:
        """Time of pruning in seconds"""
        return self._java_object.getPruningTime()

    @property
    def is_voting(self) -> bool:
        """Value indicating whether rules are voting."""
        return self._java_object.getIsVoting()

    @property
    def parameters(self) -> object:
        """Parameters used during rule set induction."""
        return InductionParameters(self._java_object.getParams())

    @property
    def stats(self) -> RuleSetStatistics:
        """Rule set statistics."""
        if self._stats is None:
            self._stats = RuleSetStatistics(self)
        return self._stats

    def covering(self, example_set) -> np.ndarray:
        """:meta private:"""
        res = []
        for rule in self.rules:
            covering_info = rule._java_object.coversUnlabelled(  # pylint: disable=protected-access
                example_set
            )
            covered_examples_indexes = []
            covered_examples_indexes += covering_info
            res.append(covered_examples_indexes)
        return np.array(res, dtype=object)

    @property
    def rules(self) -> list[Rule]:
        """List of rules objects."""
        return [Rule(java_rule) for java_rule in self._java_object.getRules()]

    def calculate_conditions_count(self) -> float:
        """
        Returns
        -------
        count: float
            Number of conditions.
        """
        return self._java_object.calculateConditionsCount()

    def calculate_induced_conditions_count(self) -> float:
        """
        Returns
        -------
        count: float
            Number of induced conditions.
        """
        return self._java_object.calculateInducedCondtionsCount()

    def calculate_avg_rule_coverage(self) -> float:
        """
        Returns
        -------
        count: float
            Average rule coverage.
        """
        return self._java_object.calculateAvgRuleCoverage()

    def calculate_avg_rule_precision(self) -> float:
        """
        Returns
        -------
        count: float
            Average rule precision.
        """
        return self._java_object.calculateAvgRulePrecision()

    def calculate_avg_rule_quality(self) -> float:
        """
        Returns
        -------
        count: float
            Average rule quality.
        """
        return self._java_object.calculateAvgRuleQuality()

    def calculate_significance(self, alpha: float) -> dict:
        """
        Parameters
        ----------
        alpha : float

        Returns
        -------
        count: float
            Significance of the rule set.
        """
        significance = self._java_object.calculateSignificance(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def calculate_significance_fdr(self, alpha: float) -> dict:
        """
        Returns
        -------
        count: dict
            Significance of the rule set with false discovery rate correction. Dictionary contains
            two fields: *fraction* (fraction of rules significant at assumed level) and *p* 
            (average p-value of all rules).
        """
        significance = self._java_object.calculateSignificanceFDR(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def calculate_significance_fwer(self, alpha: float) -> dict:
        """
        Returns
        -------
        count: dict
            Significance of the rule set with familiy-wise error rate correction. Dictionary 
            contains two fields: *fraction* (fraction of rules significant at assumed level) 
            and *p* (average p-value of all rules).
        """
        significance = self._java_object.calculateSignificanceFWER(alpha)
        return {
            'p': significance.p,
            'fraction': significance.fraction
        }

    def __str__(self):
        """Returns string representation of the object."""
        return str(self._java_object.toString())
