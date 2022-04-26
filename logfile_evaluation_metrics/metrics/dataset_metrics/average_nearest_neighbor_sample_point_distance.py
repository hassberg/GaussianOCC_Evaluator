import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors


class AverageNearestNeighborSamplePointDistance(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "average_nearest_neighbor_sample_point_distance"
        self.moi = "Sampled Points"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.ylabel(self.moi)
        title = "Average Nearest Neighbor Sampled-Point distance"
        plt.title(title)

        neigh = NearestNeighbors(n_neighbors=3)
        average_distances = []

        all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            # list contains best k runs:
            value_list = [k for i in nested_lookup(self.moi, curr_log) for subelem in i for k in subelem]
            res_list = []
            for run in value_list:
                neigh.fit(run)
                dist, _ = neigh.kneighbors(run)
                avg_run_dist = np.average([res[2] for res in dist])
                res_list.append(avg_run_dist)
            average_distances.append((model_qs, res_list))

        plt.boxplot([i[1] for i in average_distances], labels=[i[0].replace("_", "\n") for i in average_distances])
        plt.xticks(fontsize=5, rotation=340)

        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
