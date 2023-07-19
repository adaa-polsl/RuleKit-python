"""Contains classes describing rules and rulesets statistics and metrics
"""


class RuleStatistics:
    """Statistics for single rule.

    Attributes
    ----------
    p : float
        Number of positives covered by the rule (accounting weights).
    n : float
        Number of negatives covered by the rule (accounting weights).
    P : float
        Number of positives in the training set (accounting weights).
    N : float
        Number of negatives in the training set (accounting weights).
    weight : float
        Rule weight.
    pvalue : float
        Rule significance.
    """

    def __init__(self, rule):
        # pylint: disable=invalid-name
        self.p = rule.weighted_p
        self.n = rule.weighted_n
        self.P = rule.weighted_P
        self.N = rule.weighted_N
        self.weight = rule.weight
        self.pvalue = rule.pvalue

    def __str__(self):
        """Returns string representation of the object."""
        return f'(p = {self.p}, n = {self.n}, P = {self.P}, ' + \
               f'N = {self.N}, weight = {self.weight}, pvalue = {self.pvalue})'


class RuleSetStatistics:
    """Statistics for ruleset.

    Attributes
    ----------
    SIGNIFICANCE_LEVEL : float
        Significance level, default value is *0.05*


    time_total_s : float
        Time of constructing the rule set in seconds.
    time_growing_s : float
        Time of growing in seconds.
    time_pruning_s : float
        Time of pruning in seconds.
    rules_count : int
        Number of rules in ruleset.
    conditions_per_rule : float
        Average number of conditions per rule.
    induced_conditions_per_rule : float
        Average number of induced conditions.
    avg_rule_coverage : float
        Average rule coverage.
    avg_rule_precision : float
        Average rule precision.
    avg_rule_quality : float
        Average rule quality.
    pvalue : float
        rule set significance.
    FDR_pvalue : float
        Significance of the rule set with false discovery rate correction.
    FWER_pvalue : float
        Significance of the rule set with familiy-wise error rate correction.
    fraction_significant : float
        Fraction of rules significant at assumed level
    fraction_FDR_significant : float
        Fraction of rules significant, set with false discovery rate correction,
        at assumed level.
    fraction_FWER_significant : float
        Fraction of rules significant, set with familiy-wise error rate correction,
        at assumed level.
    """
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, ruleset):
        # pylint: disable=invalid-name
        self.time_total_s = ruleset.total_time
        self.time_growing_s = ruleset.growing_time
        self.time_pruning_s = ruleset.pruning_time

        self.rules_count = len(ruleset.rules)
        self.conditions_per_rule = ruleset.calculate_conditions_count()
        self.induced_conditions_per_rule = ruleset.calculate_induced_conditions_count()

        self.avg_rule_coverage = ruleset.calculate_avg_rule_coverage()
        self.avg_rule_precision = ruleset.calculate_avg_rule_precision()
        self.avg_rule_quality = ruleset.calculate_avg_rule_quality()

        self.pvalue = ruleset.calculate_significance(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.FDR_pvalue = ruleset.calculate_significance_fdr(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.FWER_pvalue = ruleset.calculate_significance_fwer(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']

        self.fraction_significant = ruleset.calculate_significance(
            RuleSetStatistics.SIGNIFICANCE_LEVEL)['fraction']
        self.fraction_FDR_significant = ruleset.calculate_significance_fdr(
            RuleSetStatistics.SIGNIFICANCE_LEVEL
        )['fraction']
        self.fraction_FWER_significant = ruleset.calculate_significance_fwer(
            RuleSetStatistics.SIGNIFICANCE_LEVEL
        )['fraction']

    def __str__(self):
        return (
            f'Time total [s]: {self.time_total_s}\n' +
            f'Time growing [s]: {self.time_growing_s}\n' +
            f'Time pruning [s]: {self.time_pruning_s}\n' +
            '\n' +
            f'Rules count: {self.rules_count}\n' +
            f'Conditions per rule: {self.conditions_per_rule}\n' +
            f'Induced conditions per rule: {self.induced_conditions_per_rule}\n' +
            '\n' +
            f'Average rule coverage: {self.avg_rule_coverage}\n' +
            f'Average rule precision: {self.avg_rule_precision}\n' +
            f'Average rule quality: {self.avg_rule_quality}\n' +
            '\n' +
            f'pvalue: {self.pvalue}\n' +
            f'FDR pvalue: {self.FDR_pvalue}\n' +
            f'FWER pvalue: {self.FWER_pvalue}\n' +
            '\n' +
            (
                f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} ' +
                f'significant: {self.fraction_significant}\n'
            ) +
            (
                f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} ' +
                f'FDR significant: {self.fraction_FDR_significant}\n'
            ) +
            (
                f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} ' +
                f'FWER significant: {self.fraction_FWER_significant}\n'
            )
        )
