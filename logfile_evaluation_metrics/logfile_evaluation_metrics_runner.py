import os
from typing import List

from matplotlib.backends.backend_pdf import PdfPages
from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric


def apply_evaluation_metric(metrics, out_path, log):
    os.makedirs(out_path, exist_ok=True)
    for metric in metrics:
        with PdfPages(os.path.join(out_path, '_eval-' + metric.name + '.pdf')) as pdf:
            metric.apply_metric(save_path=out_path, logs=log, pdf=pdf, save_fig=True) #TODO here can figs be saved..


class LogfileEvaluationMetricsRunner():

    def __init__(self, single_model_metrics: List[LogfileEvaluationMetric] = [], dataset_metric: List[LogfileEvaluationMetric] = []):
        self.single_metrics = single_model_metrics
        self.dataset_metric = dataset_metric

    def evaluate(self, logs: dict):
        for mode in logs.keys():
            for dataset in logs[mode].keys():
                apply_evaluation_metric(metrics=self.dataset_metric, out_path=os.path.join("out", mode, dataset), log=logs[mode][dataset])
                for model in logs[mode][dataset].keys():
                    for qs in logs[mode][dataset][model].keys():
                        apply_evaluation_metric(metrics=self.single_metrics, out_path=os.path.join("out", mode, dataset, model, qs),
                                                log=logs[mode][dataset][model][qs])
