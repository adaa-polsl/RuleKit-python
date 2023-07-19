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
    MEstimate = 'MEstimate'
    MutualSupport = 'MutualSupport'
    Novelty = 'Novelty'
    OddsRatio = 'OddsRatio'
    OneWaySupport = 'OneWaySupport'
    PawlakDependencyFactor = 'PawlakDependencyFactor'
    Q2 = 'Q2'
    Precision = 'Precision'
    RelativeRisk = 'RelativeRisk'
    Ripper = 'Ripper'
    RuleInterest = 'RuleInterest'
    RSS = 'RSS'
    SBayesian = 'SBayesian'
    Sensitivity = 'Sensitivity'
    Specificity = 'Specificity'
    TwoWaySupport = 'TwoWaySupport'
    WeightedLaplace = 'WeightedLaplace'
    WeightedRelativeAccuracy = 'WeightedRelativeAccuracy'
    YAILS = 'YAILS'
    LogRank = 'LogRank'


class ModelsParams(BaseModel):
    """Model for validating models hyperparameters
    """
    min_rule_covered: Optional[int] = None
    minsupp_new: Optional[int] = None
    induction_measure: Measures
    pruning_measure: Measures
    voting_measure: Measures
    max_growing: float
    enable_pruning: bool
    ignore_missing: bool
    max_uncovered_fraction: float
    select_best_candidate: bool

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
    minsupp_all: str
    max_neg2pos: float
    max_passes_count: int
    penalty_strength: float
    penalty_saturation: float
