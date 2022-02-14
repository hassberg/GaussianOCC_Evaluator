from logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages


class LogEvaluator:
    def __init__(self, metrics: [LogfileEvaluationMetric], logs: [(str, [dict])]):
        self.metrics = metrics
        self.logs = logs

    def evaluate(self):
        with PdfPages('evaluation.pdf') as pdf:
            for metric in self.metrics:
                metric.apply(self.logs, pdf)
