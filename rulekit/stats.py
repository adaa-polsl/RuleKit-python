
class RuleStatistics:

    def __init__(self, rule):
        self.p = rule.weighted_p
        self.n = rule.weighted_n
        self.P = rule.weighted_P
        self.N = rule.weighted_N
        self.weight = rule.weight
        self.pvalue = rule.pvalue

    def __str__(self):
        return f'(p = {self.p}, n = {self.n}, P = {self.P}, ' + \
               f'N = {self.N}, weight = {self.weight}, pvalue = {self.pvalue})'


class RuleSetStatistics:
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, ruleset):
        self.time_total_s = ruleset.total_time
        self.time_growing_s = ruleset.growing_time
        self.time_pruning_s = ruleset.pruning_time

        self.rules_count = len(ruleset.rules)
        self.conditions_per_rule = ruleset.calculate_conditions_count()
        self.induced_conditions_per_rule = ruleset.calculate_induced_conditions_count()

        self.avg_rule_coverage = ruleset.calculate_avg_rule_coverage()
        self.avg_rule_precision = ruleset.calculate_avg_rule_precision()
        self.avg_rule_quality = ruleset.calculate_avg_rule_quality()

        self.avg_pvalue = ruleset.calculate_significance(RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.avg_FDR_pvalue = ruleset.calculate_significance_fdr(RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']
        self.avg_FWER_pvalue = ruleset.calculate_significance_fwer(RuleSetStatistics.SIGNIFICANCE_LEVEL)['p']

        self.fraction_significant = ruleset.calculate_significance(RuleSetStatistics.SIGNIFICANCE_LEVEL)['fraction']
        self.fraction_FDR_significant = ruleset.calculate_significance_fdr(RuleSetStatistics.SIGNIFICANCE_LEVEL)[
            'fraction']
        self.fraction_FWER_significant = ruleset.calculate_significance_fwer(RuleSetStatistics.SIGNIFICANCE_LEVEL)[
            'fraction']

    def __str__(self):
        return f'Time total [s]: {self.time_total_s}\n' + \
               f'Time growing [s]: {self.time_growing_s}\n' + \
               f'Time pruning [s]: {self.time_pruning_s}\n' + \
               '\n' + \
               f'Rules count: {self.rules_count}\n' + \
               f'Conditions per rule: {self.conditions_per_rule}\n' + \
               f'Induced conditions per rule: {self.induced_conditions_per_rule}\n' + \
               '\n' + \
               f'Average rule coverage: {self.avg_rule_coverage}\n' + \
               f'Average rule precision: {self.avg_rule_precision}\n' + \
               f'Average rule quality: {self.avg_rule_quality}\n' + \
               '\n' + \
               f'Average pvalue: {self.avg_pvalue}\n' + \
               f'Average FDR pvalue: {self.avg_FDR_pvalue}\n' + \
               f'Average FWER pvalue: {self.avg_FWER_pvalue}\n' + \
               '\n' + \
               f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} significant: {self.fraction_significant}\n' + \
               f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} FDR significant: {self.fraction_FDR_significant}\n' + \
               f'Fraction {RuleSetStatistics.SIGNIFICANCE_LEVEL} FWER significant: {self.fraction_FWER_significant}\n'
