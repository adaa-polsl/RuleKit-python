"""Contains constants and classes for specyfing models parameters
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel  # pylint: disable=no-name-in-module

SURVIVAL_TIME_ATTR_ROLE: str = "survival_time"
CONTRAST_ATTR_ROLE: str = "contrast_attribute"


class Measures(Enum):
    # pylint: disable=invalid-name
    """Enum for different measures used during induction, pruning and voting.

    You can ream more about each measure and its implementation
    #41-rule-quality>`_ .
    `here <https://github.com/adaa-polsl/RuleKit/wiki/4-Quality-and-evaluation
    """
    Accuracy = 'Accuracy'
    BinaryEntropy = 'BinaryEntropy'
    C1 = 'C1'
    C2 = 'C2'
    CFoil = 'CFoil'
    CN2Significnce = 'CN2Significnce'
    Correlation = 'Correlation'
    Coverage = 'Coverage'
    FBayesianConfirmation = 'FBayesianConfirmation'
    FMeasure = 'FMeasure'
    FullCoverage = 'FullCoverage'
    GeoRSS = 'GeoRSS'
    GMeasure = 'GMeasure'
    InformationGain = 'InformationGain'
    JMeasure = 'JMeasure'
    Kappa = 'Kappa'
    Klosgen = 'Klosgen'
    Laplace = 'Laplace'
    Lift = 'Lift'
    LogicalSufficiency = 'LogicalSufficiency'
    LogRank = 'LogRank'
    MEstimate = 'MEstimate'
    MutualSupport = 'MutualSupport'
    Novelty = 'Novelty'
    OddsRatio = 'OddsRatio'
    OneWaySupport = 'OneWaySupport'
    PawlakDependencyFactor = 'PawlakDependencyFactor'
    Precision = 'Precision'
    Q2 = 'Q2'
    RelativeRisk = 'RelativeRisk'
    Ripper = 'Ripper'
    RSS = 'RSS'
    RuleInterest = 'RuleInterest'
    SBayesian = 'SBayesian'
    Sensitivity = 'Sensitivity'
    Specificity = 'Specificity'
    TwoWaySupport = 'TwoWaySupport'
    WeightedLaplace = 'WeightedLaplace'
    WeightedRelativeAccuracy = 'WeightedRelativeAccuracy'
    YAILS = 'YAILS'


DEFAULT_PARAMS_VALUE = {
    'minsupp_new': 5,
    'min_rule_covered': 5,
    'induction_measure': Measures.Correlation,
    'pruning_measure':  Measures.Correlation,
    'voting_measure': Measures.Correlation,
    'max_growing': 0.0,
    'enable_pruning': True,
    'ignore_missing': False,
    'max_uncovered_fraction': 0.0,
    'select_best_candidate': False,
    'complementary_conditions': False,
    'control_apriori_precision': True,
    'max_rule_count': 0,
    'approximate_induction': False,
    'approximate_bins_count': 100,
    'mean_based_regression': True,

    'extend_using_preferred': None,
    'extend_using_automatic': None,
    'induce_using_preferred': None,
    'induce_using_automatic': None,
    'consider_other_classes': None,
    'preferred_conditions_per_rule': None,
    'preferred_attributes_per_rule': None,

    # Contrast sets
    'minsupp_all': (0.8, 0.5, 0.2, 0.1),
    'max_neg2pos': 0.5,
    'max_passes_count': 5,
    'penalty_strength': 0.5,
    'penalty_saturation': 0.2,
}


class ModelsParams(BaseModel):
    """Model for validating models hyperparameters
    """
    min_rule_covered: Optional[int] = None
    minsupp_new: Optional[int] = DEFAULT_PARAMS_VALUE['minsupp_new']
    induction_measure: Optional[Measures] = DEFAULT_PARAMS_VALUE['induction_measure']
    pruning_measure: Optional[Measures] = DEFAULT_PARAMS_VALUE['pruning_measure']
    voting_measure: Optional[Measures] = DEFAULT_PARAMS_VALUE['voting_measure']
    max_growing: Optional[float] = DEFAULT_PARAMS_VALUE['max_growing']
    enable_pruning: Optional[bool] = DEFAULT_PARAMS_VALUE['enable_pruning']
    ignore_missing: Optional[bool] = DEFAULT_PARAMS_VALUE['ignore_missing']
    max_uncovered_fraction: Optional[float] = DEFAULT_PARAMS_VALUE['max_uncovered_fraction']
    select_best_candidate: Optional[bool] = DEFAULT_PARAMS_VALUE['select_best_candidate']
    complementary_conditions: Optional[bool] = DEFAULT_PARAMS_VALUE['complementary_conditions']
    max_rule_count: int = DEFAULT_PARAMS_VALUE['max_rule_count']
    minsupp_all: Optional[str] = None

    extend_using_preferred: Optional[bool] = None
    extend_using_automatic: Optional[bool] = None
    induce_using_preferred: Optional[bool] = None
    induce_using_automatic: Optional[bool] = None
    consider_other_classes: Optional[bool] = None
    preferred_conditions_per_rule: Optional[int] = None
    preferred_attributes_per_rule: Optional[int] = None


class ContrastSetModelParams(ModelsParams):
    """Model for validating contrast set models hyperparameters
    """
    minsupp_all: Optional[str] = DEFAULT_PARAMS_VALUE['minsupp_all']
    max_neg2pos: Optional[float] = DEFAULT_PARAMS_VALUE['max_neg2pos']
    max_passes_count: Optional[int] = DEFAULT_PARAMS_VALUE['max_passes_count']
    penalty_strength: Optional[float] = DEFAULT_PARAMS_VALUE['penalty_strength']
    penalty_saturation: Optional[float] = DEFAULT_PARAMS_VALUE['penalty_saturation']
