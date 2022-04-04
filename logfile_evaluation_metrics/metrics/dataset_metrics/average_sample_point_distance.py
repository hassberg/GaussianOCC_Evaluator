import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class AverageSamplePointDistance(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "average_sample_point_distance"
        self.moi = "Sampled Points"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.ylabel(self.moi)
        title = "Average Sampled-Point distance"
        plt.title(title)

        average_distances = []

        all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            # list contains best k runs:
            value_list = [k for i in nested_lookup(self.moi, curr_log) for subelem in i for k in subelem]
            average_distances.append((model_qs, list(map(lambda x: np.mean(pdist(x)), value_list))))

        plt.boxplot([i[1] for i in average_distances], labels=[i[0].replace("_", "\n") for i in average_distances])
        plt.xticks(fontsize=6, rotation=10)

        plt.legend(fontsize=4)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
