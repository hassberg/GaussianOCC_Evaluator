from typing import List

from matplotlib.backends.backend_pdf import PdfPages
from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric


class LogfileEvaluationMetricsRunner():
    def __init__(self, filename: str = "Evaluation"):
        self.pdf = PdfPages(filename + '.pdf')

    def evaluate(self, metrics_name: str, metrics: List[LogfileEvaluationMetric], logs: dict):
        for metric in metrics:
            metric.apply_metric(metrics_name, logs, self.pdf)

    def end(self):
        self.pdf.close()
