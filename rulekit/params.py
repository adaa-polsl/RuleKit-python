"""Contains constants and classes for specyfing models parameters
"""
from enum import Enum
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

from jpype import JImplements
from jpype import JOverride
from jpype.types import JDouble
from pydantic import BaseModel  # pylint: disable=no-name-in-module

MAX_INT: int = 2147483647  # max integer value in Java

_UserDefinedMeasure = Callable[[float, float, float, float], float]


def _user_defined_measure_factory(measure_function: _UserDefinedMeasure):
    from adaa.analytics.rules.logic.quality import \
        IUserMeasure  # pylint: disable=import-outside-toplevel,import-error

    @JImplements(IUserMeasure)
    class _UserMeasure:  # pylint: disable=invalid-name,missing-function-docstring

        @JOverride
        def getResult(self, p: JDouble, n: JDouble, P: JDouble, N: JDouble) -> float:
            return measure_function(float(p), float(n), float(P), float(N))

    return _UserMeasure()


class Measures(Enum):
    # pylint: disable=invalid-name
    """Enum for different measures used during induction, pruning and voting.

    You can ream more about each measure and its implementation
    #41-rule-quality>`_ .
    `here <https://github.com/adaa-polsl/RuleKit/wiki/4-Quality-and-evaluation
    """
    Accuracy = "Accuracy"
    BinaryEntropy = "BinaryEntropy"
    C1 = "C1"
    C2 = "C2"
    CFoil = "CFoil"
    CN2Significnce = "CN2Significnce"
    Correlation = "Correlation"
    Coverage = "Coverage"
    FBayesianConfirmation = "FBayesianConfirmation"
    FMeasure = "FMeasure"
    FullCoverage = "FullCoverage"
    GeoRSS = "GeoRSS"
    GMeasure = "GMeasure"
    InformationGain = "InformationGain"
    JMeasure = "JMeasure"
    Kappa = "Kappa"
    Klosgen = "Klosgen"
    Laplace = "Laplace"
    Lift = "Lift"
    LogicalSufficiency = "LogicalSufficiency"
    LogRank = "LogRank"
    MEstimate = "MEstimate"
    MutualSupport = "MutualSupport"
    Novelty = "Novelty"
    OddsRatio = "OddsRatio"
    OneWaySupport = "OneWaySupport"
    PawlakDependencyFactor = "PawlakDependencyFactor"
    Precision = "Precision"
    Q2 = "Q2"
    RelativeRisk = "RelativeRisk"
    Ripper = "Ripper"
    RSS = "RSS"
    RuleInterest = "RuleInterest"
    SBayesian = "SBayesian"
    Sensitivity = "Sensitivity"
    Specificity = "Specificity"
    TwoWaySupport = "TwoWaySupport"
    WeightedLaplace = "WeightedLaplace"
    WeightedRelativeAccuracy = "WeightedRelativeAccuracy"
    YAILS = "YAILS"


DEFAULT_PARAMS_VALUE = {
    "minsupp_new": 0.05,
    "induction_measure": Measures.Correlation,
    "pruning_measure": Measures.Correlation,
    "voting_measure": Measures.Correlation,
    "max_growing": 0.0,
    "enable_pruning": True,
    "ignore_missing": False,
    "max_uncovered_fraction": 0.0,
    "select_best_candidate": False,
    "complementary_conditions": False,
    "control_apriori_precision": True,
    "max_rule_count": 0,
    "approximate_induction": False,
    "approximate_bins_count": 100,
    "mean_based_regression": True,
    "extend_using_preferred": False,
    "extend_using_automatic": False,
    "induce_using_preferred": False,
    "induce_using_automatic": False,
    "consider_other_classes": False,
    "preferred_conditions_per_rule": MAX_INT,
    "preferred_attributes_per_rule": MAX_INT,
    # Contrast sets
    "minsupp_all": (0.8, 0.5, 0.2, 0.1),
    "max_neg2pos": 0.5,
    "max_passes_count": 5,
    "penalty_strength": 0.5,
    "penalty_saturation": 0.2,
}

_QualityMeasure = Union[Measures, _UserDefinedMeasure]


class ModelsParams(BaseModel):
    """Model for validating models hyperparameters"""

    minsupp_new: Optional[float] = DEFAULT_PARAMS_VALUE["minsupp_new"]
    induction_measure: Optional[_QualityMeasure] = DEFAULT_PARAMS_VALUE[
        "induction_measure"
    ]
    pruning_measure: Optional[_QualityMeasure] = DEFAULT_PARAMS_VALUE["pruning_measure"]
    voting_measure: Optional[_QualityMeasure] = DEFAULT_PARAMS_VALUE["voting_measure"]
    max_growing: Optional[float] = DEFAULT_PARAMS_VALUE["max_growing"]
    enable_pruning: Optional[bool] = DEFAULT_PARAMS_VALUE["enable_pruning"]
    ignore_missing: Optional[bool] = DEFAULT_PARAMS_VALUE["ignore_missing"]
    max_uncovered_fraction: Optional[float] = DEFAULT_PARAMS_VALUE[
        "max_uncovered_fraction"
    ]
    select_best_candidate: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "select_best_candidate"
    ]
    complementary_conditions: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "complementary_conditions"
    ]
    max_rule_count: int = DEFAULT_PARAMS_VALUE["max_rule_count"]


class ExpertModelParams(ModelsParams):
    """Model for validating expert models hyperparameters"""

    extend_using_preferred: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "extend_using_preferred"
    ]
    extend_using_automatic: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "extend_using_automatic"
    ]
    induce_using_preferred: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "induce_using_preferred"
    ]
    induce_using_automatic: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "induce_using_automatic"
    ]
    consider_other_classes: Optional[bool] = DEFAULT_PARAMS_VALUE[
        "consider_other_classes"
    ]
    preferred_conditions_per_rule: Optional[int] = DEFAULT_PARAMS_VALUE[
        "preferred_conditions_per_rule"
    ]
    preferred_attributes_per_rule: Optional[int] = DEFAULT_PARAMS_VALUE[
        "preferred_attributes_per_rule"
    ]


class ContrastSetModelParams(ModelsParams):
    """Model for validating contrast set models hyperparameters"""

    minsupp_all: Tuple[float, float, float, float] = DEFAULT_PARAMS_VALUE["minsupp_all"]
    max_neg2pos: Optional[float] = DEFAULT_PARAMS_VALUE["max_neg2pos"]
    max_passes_count: Optional[int] = DEFAULT_PARAMS_VALUE["max_passes_count"]
    penalty_strength: Optional[float] = DEFAULT_PARAMS_VALUE["penalty_strength"]
    penalty_saturation: Optional[float] = DEFAULT_PARAMS_VALUE["penalty_saturation"]
