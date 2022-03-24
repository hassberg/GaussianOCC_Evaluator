import os
from typing import List

from matplotlib.backends.backend_pdf import PdfPages
from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric


class LogfileEvaluationMetricsRunner():

    def evaluate(self, metrics_name: str, metrics: List[LogfileEvaluationMetric], logs: dict):
        for metric in metrics:
            with PdfPages(os.path.join("out",'eval-' + metric.name + '.pdf')) as pdf:
                metric.apply_metric(metrics_name, logs, pdf)
