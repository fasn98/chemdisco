"""QSAR modelling, with the checks that make a reported metric mean something."""

from .applicability import ApplicabilityDomain, DomainVerdict
from .baselines import (
    BaselineScores,
    compute_baselines,
    detect_target_leakage,
    nearest_neighbour_predictor,
    r_squared,
    rmse,
)
from .evaluate import (
    Evaluation,
    Interval,
    SplitComparison,
    bootstrap_metric,
    evaluate,
    mae,
    pearson,
    spearman,
)
from .model import DEFAULT_RANDOM_STATE, QSARModel, make_fit_predict

__all__ = [
    "ApplicabilityDomain",
    "BaselineScores",
    "DEFAULT_RANDOM_STATE",
    "DomainVerdict",
    "Evaluation",
    "Interval",
    "QSARModel",
    "SplitComparison",
    "bootstrap_metric",
    "compute_baselines",
    "detect_target_leakage",
    "evaluate",
    "mae",
    "make_fit_predict",
    "nearest_neighbour_predictor",
    "pearson",
    "r_squared",
    "rmse",
    "spearman",
]

from .dataset import (  # noqa: E402  (re-exported after the list above)
    LabelProvenanceError,
    TrainingSet,
    build_training_set,
    labels_from_points,
    verify_label_provenance,
)

__all__ += [
    "LabelProvenanceError",
    "TrainingSet",
    "build_training_set",
    "labels_from_points",
    "verify_label_provenance",
]
