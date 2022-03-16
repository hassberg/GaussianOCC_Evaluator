import os
import json

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor
from log_evalmetric_extractor.matthew_correlation_coefficient_extractor import MatthewCorrelationCoefficientExtractor
from logfile_evaluation_metrics.logfile_evaluation_metrics_runner import LogfileEvaluationMetricsRunner
from logfile_evaluation_metrics.metrics.all_learning_curves_plot import AllLearningCurvesPlot
from logfile_evaluation_metrics.metrics.average_learning_curve_plot import AverageLearningCurvePlot
from logfile_evaluation_metrics.metrics.averaged_quality_range import AverageQualityRange


def get_logged_metric(file_list, eval_metric_extractor: EvalMetricExtractor) -> dict:
    log_metric_dictionary = {}

    for file in file_list:
        path_array = file.split(os.sep)
        key = path_array[1] + "_" + path_array[2]

        file_content = open(file)
        data_object = json.loads(file_content.read())
        new_values = eval_metric_extractor.get_metrics_log(data_object)

        if key in log_metric_dictionary:
            log_metric_dictionary[key].extend(new_values)
        else:
            log_metric_dictionary[key] = new_values

    return log_metric_dictionary


metrics_extractors = [MatthewCorrelationCoefficientExtractor()]
log_evaluation_metrics = [AllLearningCurvesPlot(), AverageLearningCurvePlot(), AverageQualityRange()]

log_eval_runner = LogfileEvaluationMetricsRunner()

root = "logfiles"
files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

for metrics_extractor in metrics_extractors:
    full_metrics_scoring = get_logged_metric(files, metrics_extractor)
    log_eval_runner.evaluate(metrics_extractor.name, log_evaluation_metrics, full_metrics_scoring)

log_eval_runner.end()
