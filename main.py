from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor
from log_evalmetric_extractor.logfile_reader import read_log
from log_evalmetric_extractor.metric_extractors.certainty_to_missclassification import UncertaintyMisclassificationCorrelation
from log_evalmetric_extractor.metric_extractors.hyperparameter_extactor import HyperparameterSelected
from log_evalmetric_extractor.metric_extractors.matthew_correlation_coefficient_extractor import MatthewCorrelationCoefficientExtractor
from log_evalmetric_extractor.metric_extractors.mcc_sample_extractor import MccSampleExtractor
from log_evalmetric_extractor.metric_extractors.relative_certainty_to_missclassification import RelativeCertaintyMisclassificationCorrelation
from log_evalmetric_extractor.metric_extractors.sampled_label import SampledLabel
from log_evalmetric_extractor.metric_extractors.sampled_point import SampledPoints
from log_evalmetric_extractor.metric_extractors.sampled_point_with_label import SampledPointsWithLabel
from log_evalmetric_extractor.metric_extractors.uncertainty_confusion_correlation_extractor import UncertaintyConfusionCorrelationExtractor
from log_evalmetric_extractor.metric_extractors.weigthed_mcc_extractor import WeightedMccExtractor
from logfile_evaluation_metrics.logfile_evaluation_metrics_runner import LogfileEvaluationMetricsRunner
from logfile_evaluation_metrics.metrics.dataset_metrics.average_best_learning_curve_plot_with_error_bar import AverageBestLearningCurvePlotWithStd
from logfile_evaluation_metrics.metrics.dataset_metrics.average_nearest_neighbor_sample_point_distance import AverageNearestNeighborSamplePointDistance
from logfile_evaluation_metrics.metrics.dataset_metrics.average_outlier_sampling_ratio import AverageOutlierSamplingRatio
from logfile_evaluation_metrics.metrics.dataset_metrics.average_sample_point_distance import AverageSamplePointDistance
from logfile_evaluation_metrics.metrics.single_model_metric.all_learning_curves_plot import AllLearningCurvesPlot
from logfile_evaluation_metrics.metrics.dataset_metrics.average_best_learning_curve_plot import AverageBestLearningCurvePlot
from logfile_evaluation_metrics.metrics.dataset_metrics.average_learning_curve_plot import AverageLearningCurvePlot
from logfile_evaluation_metrics.metrics.dataset_metrics.average_learning_curve_with_std import AverageLearningCurveWithStd
from logfile_evaluation_metrics.metrics.dataset_metrics.averaged_quality_range import AverageQualityRange
from logfile_evaluation_metrics.metrics.improvement_vs_outlier import ImprovementVsOutlier
from logfile_evaluation_metrics.metrics.dataset_metrics.learning_curve_outlier_sampled_dropout import AverageLearningCurveDropoutOutlierSampled
from logfile_evaluation_metrics.metrics.single_model_metric.average_learning_curve_dropout_range_with_std import AverageLearningCurveByDropoutRangeWithStd
from logfile_evaluation_metrics.metrics.single_model_metric.certainty_correctness_eval import CertaintyCorrectnessEval
from logfile_evaluation_metrics.metrics.single_model_metric.hyperparameter_eval_metric import HyperparameterEvalMetric
from logfile_evaluation_metrics.metrics.single_model_metric.learning_curve_vs_sample import LearningCurveVsSample
from logfile_evaluation_metrics.metrics.outlier_sampling import OutlierSampling
from logfile_evaluation_metrics.metrics.single_model_metric.relative_certainty_correctness_eval import RelativeCertaintyCorrectnessEval
from logfile_evaluation_metrics.metrics.single_model_metric.sample_point_distance_by_hyperparameter_selection import SamplePointDistancesByHyperparameterSelection
from logfile_evaluation_metrics.metrics.single_model_metric.uncertainty_confusion_dev import UncertaintyConfusionDev
from logfile_evaluation_metrics.metrics.single_model_metric.weighted_mcc import WeightedMcc

root = "logfiles"
extractors = [SampledPointsWithLabel(),SampledLabel(), SampledPoints(), WeightedMccExtractor(), UncertaintyConfusionCorrelationExtractor(), RelativeCertaintyMisclassificationCorrelation(),UncertaintyMisclassificationCorrelation(), HyperparameterSelected(), MatthewCorrelationCoefficientExtractor(), MccSampleExtractor()]
logged_dict = read_log(root, extractors)

single_model_metric = [SampledPointsWithLabel(), WeightedMcc(), RelativeCertaintyCorrectnessEval(), CertaintyCorrectnessEval(), UncertaintyConfusionDev(), LearningCurveVsSample(),
                       SamplePointDistancesByHyperparameterSelection(), HyperparameterEvalMetric(), AverageLearningCurveByDropoutRangeWithStd()]
dataset_metric = [AverageOutlierSamplingRatio(), AverageNearestNeighborSamplePointDistance(), AverageLearningCurvePlot(), AverageBestLearningCurvePlot(),
                  AverageBestLearningCurvePlotWithStd(), AverageQualityRange(),
                  AverageLearningCurveDropoutOutlierSampled(), AverageSamplePointDistance(), AverageLearningCurveWithStd()]

print("on Dataset-Metrics")
for metric in dataset_metric:
    print("started: " + metric.name)
    log_eval_runner = LogfileEvaluationMetricsRunner(single_model_metrics=[], dataset_metric=[metric])
    log_eval_runner.evaluate(logged_dict)
    print("finished: " + metric.name)

print("on SingleModel-Metrics")
for metric in single_model_metric:
    print("started: " + metric.name)
    log_eval_runner = LogfileEvaluationMetricsRunner(single_model_metrics=[metric], dataset_metric=[])
    log_eval_runner.evaluate(logged_dict)
    print("finished: " + metric.name)
