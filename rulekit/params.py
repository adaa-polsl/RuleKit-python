from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Measures(Enum):
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
    min_rule_covered: Optional[int]
    minsupp_new: Optional[int]
    induction_measure: Measures
    pruning_measure: Measures
    voting_measure: Measures
    max_growing: float
    enable_pruning: bool
    ignore_missing: bool
    max_uncovered_fraction: float
    select_best_candidate: bool

    extend_using_preferred: Optional[bool]
    extend_using_automatic: Optional[bool]
    induce_using_preferred: Optional[bool]
    induce_using_automatic: Optional[bool]
    consider_other_classes: Optional[bool]
