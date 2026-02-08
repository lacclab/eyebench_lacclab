from dataclasses import dataclass, field

from src.configs.constants import (
    BackboneNames,
    ItemLevelFeaturesModes,
    MLModelNames,
)
from src.configs.models.base_model import MLModelArgs
from src.configs.utils import register_model_config


# =============================================================================
# BASE LOGISTIC REGRESSION MODEL
# =============================================================================

@register_model_config
@dataclass
class LogisticRegressionMLArgs(MLModelArgs):
    """
    Model arguments for the Logistic Regression model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_clf__C (float): Inverse of regularization strength.
        sklearn_pipeline_param_clf__fit_intercept (bool): Whether to add an intercept to the decision function.
        sklearn_pipeline_param_clf__penalty (str): Norm used in penalization.
        sklearn_pipeline_param_clf__solver (str): Optimization algorithm.
        sklearn_pipeline_param_clf__random_state (int): Seed for pseudo-random number generator.
        sklearn_pipeline_param_clf__max_iter (int): Maximum number of solver iterations.
        sklearn_pipeline_param_clf__class_weight (str): Class weight balancing strategy.
        sklearn_pipeline_param_scaler__with_mean (bool): Whether to center data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): Whether to scale data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.LOGISTIC_REGRESSION

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('clf', 'sklearn.linear_model.LogisticRegression'),
    )
    sklearn_pipeline_param_clf__C: float = 2.0
    sklearn_pipeline_param_clf__fit_intercept: bool = True
    sklearn_pipeline_param_clf__penalty: str = 'l2'
    sklearn_pipeline_param_clf__solver: str = 'lbfgs'
    sklearn_pipeline_param_clf__random_state: int = 1
    sklearn_pipeline_param_clf__max_iter: int = 1000
    sklearn_pipeline_param_clf__class_weight: str = 'balanced'
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED],
    )


# =============================================================================
# LOGISTIC REGRESSION - SINGLE FEATURE MODELS
# =============================================================================

@register_model_config
@dataclass
class LogisticMeziereArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )


@register_model_config
@dataclass
class LogisticFixationMetricsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.FIXATION_METRICS],
    )


@register_model_config
@dataclass
class LogisticSClustersArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.S_CLUSTERS],
    )


@register_model_config
@dataclass
class LogisticSClustersNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM],
    )


@register_model_config
@dataclass
class LogisticWpCoefsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS],
    )


@register_model_config
@dataclass
class LogisticWpCoefsNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_NORM],
    )


@register_model_config
@dataclass
class LogisticWpCoefsNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT],
    )


@register_model_config
@dataclass
class LogisticWpCoefsNoNormNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT],
    )


# =============================================================================
# LOGISTIC REGRESSION - TWO FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedWpCoefsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedWpCoefsNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedWpCoefsNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedWpCoefsNoNormNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


# =============================================================================
# LOGISTIC REGRESSION - THREE FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsWpCoefsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsWpCoefsNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsWpCoefsNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsWpCoefsNoNormNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersWpCoefsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersNoNormWpCoefsNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersWpCoefsNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedSClustersNoNormWpCoefsNoNormNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


# =============================================================================
# LOGISTIC REGRESSION - FOUR FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersWpCoefsArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersWpCoefsNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LogisticReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormNoInterceptArgs(LogisticRegressionMLArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


# =============================================================================
# BASE LINEAR REGRESSION MODEL
# =============================================================================

@register_model_config
@dataclass
class LinearRegressionArgs(MLModelArgs):
    """
    Model arguments for the Linear Regression model.

    Attributes:
        batch_size (int): The batch size for training.
        use_fixation_report (bool): Whether to use the fixation report.
        backbone (str): The backbone model to use.
        sklearn_pipeline (tuple): The scikit-learn pipeline for the model.
        sklearn_pipeline_param_regressor__fit_intercept (bool): Whether to calculate the intercept for this model.
        sklearn_pipeline_param_scaler__with_mean (bool): Whether to center data before scaling.
        sklearn_pipeline_param_scaler__with_std (bool): Whether to scale data to unit variance.
    """

    base_model_name: MLModelNames = MLModelNames.LINEAR_REG

    sklearn_pipeline: tuple = (
        ('scaler', 'sklearn.preprocessing.StandardScaler'),
        ('regressor', 'sklearn.linear_model.LinearRegression'),
    )
    sklearn_pipeline_param_regressor__fit_intercept: bool = True
    sklearn_pipeline_param_scaler__with_mean: bool = True
    sklearn_pipeline_param_scaler__with_std: bool = True

    batch_size: int = 1024
    use_fixation_report: bool = True
    backbone: BackboneNames = BackboneNames.XLM_ROBERTA_LARGE
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.READING_SPEED],
    )


# =============================================================================
# LINEAR REGRESSION - SINGLE FEATURE MODELS
# =============================================================================

@register_model_config
@dataclass
class LinearMeziereArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.LOGISTIC],
    )


@register_model_config
@dataclass
class LinearFixationMetricsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.FIXATION_METRICS],
    )


@register_model_config
@dataclass
class LinearSClustersArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.S_CLUSTERS],
    )


@register_model_config
@dataclass
class LinearSClustersNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM],
    )


@register_model_config
@dataclass
class LinearWpCoefsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS],
    )


@register_model_config
@dataclass
class LinearWpCoefsNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_NORM],
    )


@register_model_config
@dataclass
class LinearWpCoefsNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT],
    )


@register_model_config
@dataclass
class LinearWpCoefsNoNormNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT],
    )


# =============================================================================
# LINEAR REGRESSION - TWO FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedWpCoefsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedWpCoefsNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedWpCoefsNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedWpCoefsNoNormNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


# =============================================================================
# LINEAR REGRESSION - THREE FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsWpCoefsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsWpCoefsNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsWpCoefsNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsWpCoefsNoNormNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersWpCoefsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersNoNormWpCoefsNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersWpCoefsNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedSClustersNoNormWpCoefsNoNormNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )


# =============================================================================
# LINEAR REGRESSION - FOUR FEATURE COMBINATIONS
# =============================================================================

@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersWpCoefsArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersWpCoefsNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS,
            ItemLevelFeaturesModes.WP_COEFS_NO_INTERCEPT,
        ],
    )


@register_model_config
@dataclass
class LinearReadingSpeedFixationMetricsSClustersNoNormWpCoefsNoNormNoInterceptArgs(LinearRegressionArgs):
    item_level_features_modes: list[ItemLevelFeaturesModes] = field(
        default_factory=lambda: [
            ItemLevelFeaturesModes.READING_SPEED,
            ItemLevelFeaturesModes.FIXATION_METRICS,
            ItemLevelFeaturesModes.S_CLUSTERS_NO_NORM,
            ItemLevelFeaturesModes.WP_COEFS_NO_NORM_NO_INTERCEPT,
        ],
    )

