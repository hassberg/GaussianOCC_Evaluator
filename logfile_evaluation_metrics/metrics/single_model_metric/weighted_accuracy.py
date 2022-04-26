import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class WeightedAccuracy(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "weighted_accuracy_comparison"
        self.moi = "Weighted Accuracy"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        if not "SVDD" in save_path:
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            title = "Weighted vs Unweighted Accuracy"
            plt.title(title)

            value_list = [i for sublist in nested_lookup(self.moi, logs) for repeats in sublist for i in repeats]

            iter = []
            for run in value_list:
                if iter == []:
                    iter = [[i] for i in run]

                else:
                    for step in range(len(run)):
                        iter[step].append(run[step])

            labels = ["Weighted Accuracy", "Unweighted Accuracy"]
            for type in range(len(labels)):
                steping = []
                for i in range(len(iter)):
                    steping.append([step[type] for step in iter[i]])

                mean = np.average(steping, axis=1)
                std = np.std(steping, axis=1)
                plt.plot(range(len(mean)), mean, label=labels[type])
                plt.fill_between(range(len(mean)), mean + std, mean - std, alpha=0.3)

            plt.legend(fontsize=4)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            pdf.savefig()
            plt.close()
