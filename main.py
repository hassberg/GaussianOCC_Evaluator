import os
import json
from json import JSONDecodeError

from nested_lookup import nested_lookup
from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor
from log_evalmetric_extractor.logfile_reader import read_log
from log_evalmetric_extractor.metric_extractors.matthew_correlation_coefficient_extractor import MatthewCorrelationCoefficientExtractor
from log_evalmetric_extractor.metric_extractors.mcc_sample_extractor import MccSampleExtractor
from logfile_evaluation_metrics.logfile_evaluation_metrics_runner import LogfileEvaluationMetricsRunner
from logfile_evaluation_metrics.metrics.all_learning_curves_plot import AllLearningCurvesPlot
from logfile_evaluation_metrics.metrics.average_best_learning_curve_plot import AverageBestLearningCurvePlot
from logfile_evaluation_metrics.metrics.average_learning_curve_plot import AverageLearningCurvePlot
from logfile_evaluation_metrics.metrics.average_learning_curve_with_error import AverageLearningCurveWithErrorBar
from logfile_evaluation_metrics.metrics.averaged_quality_range import AverageQualityRange
from logfile_evaluation_metrics.metrics.custom_parameter import CustomParameter
from logfile_evaluation_metrics.metrics.improvement_vs_outlier import ImprovementVsOutlier
from logfile_evaluation_metrics.metrics.learning_curve_outlier_sampled_dropout import LearningCurveOutlierSampled
from logfile_evaluation_metrics.metrics.learning_curve_vs_sample import LearningCurveVsSample
from logfile_evaluation_metrics.metrics.outlier_sampling import OutlierSampling
from logfile_evaluation_metrics.metrics.trainable_progress import TrainableProgress
from logfile_evaluation_metrics.metrics.vanishing_progress import VanishingProgress





root = "logfiles"
extractors = [MatthewCorrelationCoefficientExtractor(), MccSampleExtractor()]
logged_dict = read_log(root, extractors)

log_eval_runner = LogfileEvaluationMetricsRunner()

# mcc_log_evaluation_metrics = [AllLearningCurvesPlot(), AverageLearningCurvePlot(), AverageQualityRange(), AverageBestLearningCurvePlot()]
# mcc_extractor = MatthewCorrelationCoefficientExtractor()
# full_metrics_scoring = get_logged_metric(files, mcc_extractor)
# log_eval_runner.evaluate(mcc_extractor.name, mcc_log_evaluation_metrics, full_metrics_scoring)

# log_eval_runner.evaluate("Outlier Sampled", [LearningCurveOutlierSampled()], get_logged_metric(files, MccSampleExtractor()))
# log_eval_runner.evaluate("Outlier Sampled", [OutlierSampling(),ImprovementVsOutlier()], get_logged_metric(files, MccSampleExtractor()))

# log_eval_runner.evaluate("Vanishable Prior", [VanishingProgress()], get_log_for_model(files, "VanishingSelfTrainingCustomModelBasedPriorMeanSurrogateModel"))
# log_eval_runner.evaluate("Trainable Lengthscale", [TrainableProgress()], get_log_for_model(files, "SelfTrainingCustomModelBasedPriorMeanSurrogateModel"))
# log_eval_runner.evaluate("Custom and Fix", [CustomParameter()], get_log_for_model(files, "CustomModelBasedPriorMeanSurrogateModel"))
