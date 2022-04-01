from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class ImprovementVsOutlier(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "outlier_vs_improvement"

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for key, values in logs.items():
            plt.xlabel("Outlier")
            plt.ylabel("End Quality")

            samples = [i[1] for subelems in values.items() for i in subelems[1]]
            mcc = [i[0] for subelems in values.items() for i in subelems[1]]

            outlier_sampled = list(map(lambda x: sum(x), list(map(lambda x: list(map(lambda y: float(y) * -0.5 + 0.5, x)), samples))))
            mcc_improvement = list(map(lambda x: x[len(x) - 1], mcc))

            plt.scatter(outlier_sampled, mcc_improvement)

            plt.title(key.replace("_", "\n"))
            plt.savefig("outlier_vs_imporvement-" + key + ".svg")
            pdf.savefig()
            plt.close()
