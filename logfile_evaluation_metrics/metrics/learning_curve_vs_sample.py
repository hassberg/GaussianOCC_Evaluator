from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveVsSample(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "learning_curve_vs_sampled"

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for key, values in logs.items():
            scorings = [i[0] for subelems in values.items() for i in subelems[1]]
            samples = [i[1] for subelems in values.items() for i in subelems[1]]

            fig1, ax = plt.subplots()
            ax.plot(range(len(scorings[0])), np.average(scorings, axis=0))
            ax.plot([i + 1 for i in range(len(samples[0]))], np.average(samples, axis=0))

            ax.set_title(key.split("_")[0] + "\n" + key.split("_")[1])
            pdf.savefig(fig1)
            plt.close(fig1)
