import os
import json

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor
from log_evalmetric_extractor.matthew_correlation_coefficient_extractor import MatthewCorrelationCoefficientExtractor
from log_evalmetric_extractor.mcc_sample_extractor import MccSampleExtractor
from logfile_evaluation_metrics.logfile_evaluation_metrics_runner import LogfileEvaluationMetricsRunner
from logfile_evaluation_metrics.metrics.all_learning_curves_plot import AllLearningCurvesPlot
from logfile_evaluation_metrics.metrics.average_best_learning_curve_plot import AverageBestLearningCurvePlot
from logfile_evaluation_metrics.metrics.average_learning_curve_plot import AverageLearningCurvePlot
from logfile_evaluation_metrics.metrics.average_learning_curve_with_error import AverageLearningCurveWithErrorBar
from logfile_evaluation_metrics.metrics.averaged_quality_range import AverageQualityRange
from logfile_evaluation_metrics.metrics.learning_curve_vs_sample import LearningCurveVsSample


def get_logged_metric(file_list, eval_metric_extractor: EvalMetricExtractor) -> dict:
    log_metric_dictionary = {}

    for file in file_list:
        path_array = file.split(os.sep)
        key = path_array[1] + "_" + path_array[2]
        data_sample = path_array[3].split('_')[0] + "-sample" + path_array[3].split('_')[1]

        file_content = open(file)
        data_object = json.loads(file_content.read())
        new_values = eval_metric_extractor.get_metrics_log(data_object)

        if key in log_metric_dictionary:
            if data_sample in log_metric_dictionary[key]:
                log_metric_dictionary[key][data_sample].extend(new_values)
            else:
                log_metric_dictionary[key][data_sample] = new_values
        else:
            log_metric_dictionary[key] = {data_sample: new_values}

    return log_metric_dictionary


log_eval_runner = LogfileEvaluationMetricsRunner()

root = "logfiles"
files = [os.path.join(path, name) for path, subdirs, files in os.walk(root) for name in files]

mcc_log_evaluation_metrics = [AllLearningCurvesPlot(), AverageLearningCurvePlot(), AverageQualityRange(), AverageLearningCurveWithErrorBar(),AverageBestLearningCurvePlot()]
mcc_extractor = MatthewCorrelationCoefficientExtractor()
full_metrics_scoring = get_logged_metric(files, mcc_extractor)
log_eval_runner.evaluate(mcc_extractor.name, mcc_log_evaluation_metrics, full_metrics_scoring)

log_eval_runner.evaluate("mcc and sample", [LearningCurveVsSample()], get_logged_metric(files, MccSampleExtractor()))
