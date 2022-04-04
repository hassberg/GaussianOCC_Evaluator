import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveVsSample(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "learning_curve_vs_sampled"
        self.moi = "Mcc vs Sample"

    def apply_metric(self, save_path: str, logs: dict, pdf: PdfPages, save_fig: bool = False):
        fig1, ax = plt.subplots()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mcc")
        title = "Average Outlier Sampled vs Mcc"

        ax2 = ax.twinx()
        ax2.set_ylabel("Avg Outlier Sampled")
        ax.set_title(title)

        value_list = [k for i in nested_lookup(self.moi, logs) for subelem in i for k in subelem]

        avg_mcc = np.average([i[0] for i in value_list], axis=0)
        ax.plot(range(len(avg_mcc)), avg_mcc, label="mcc")

        avg_sample = np.average([i[1] for i in value_list], axis=0)
        ax2.plot([i + 1 for i in range(len(avg_sample))], avg_sample, '--', label="sample")

        pdf.savefig(fig1)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))

        plt.close(fig1)
