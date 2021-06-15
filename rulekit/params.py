from enum import Enum


class Measures(Enum):
    """Enum for different measures used during induction, pruning and voting.

    You can ream more about each measure and its implementation
    `here <https://github.com/adaa-polsl/RuleKit/wiki/4-Quality-and-evaluation#41-rule-quality>`_ . 
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
