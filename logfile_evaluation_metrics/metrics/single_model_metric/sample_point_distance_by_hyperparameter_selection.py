import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class SamplePointDistancesByHyperparameterSelection(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "sample_point_distance_by_hyperparameter_selection"
        self.moi = "Sampled Points"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.ylabel(self.moi)
        title = "Sampled-Points distance by Hyperparameter Selection"
        plt.title(title)

        average_distances = []

        lookup = nested_lookup(self.moi, logs)
        for k in range(len(lookup)):
            sampled_points = [rd_points for repeats in lookup[k] for rd_points in repeats]
            average_distances.append((k, list(map(lambda x: np.mean(pdist(x)), sampled_points))))

        plt.boxplot([i[1] for i in average_distances], labels=[str(i[0] + 1) for i in average_distances])
        plt.xticks(fontsize=6, rotation=10)

        plt.legend(fontsize=4)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
