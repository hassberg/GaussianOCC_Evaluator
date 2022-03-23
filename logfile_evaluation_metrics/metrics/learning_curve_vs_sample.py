from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveVsSample(LogfileEvaluationMetric):
    def __init__(self, ):
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        plt.xlabel("Iterations")
        plt.ylabel(metrics_name)
        title = "Average Learning Curve vs samples"
        plt.title(title)
        for key, values in logs.items():
            for rd in [i for subelems in values.items() for i in subelems[1]]:
                plt.plot(range(len(rd[0])), rd[0])
                plt.plot([i+1 for i in range(len(rd[1]))], rd[1])

                # plt.legend(fontsize=5)

                # plt.savefig("Average_lerning_curve_dropout-" + str(dropout) + ".svg")
                pdf.savefig()
                plt.close()
