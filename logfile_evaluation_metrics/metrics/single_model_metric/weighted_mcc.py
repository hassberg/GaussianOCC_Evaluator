import os

from nested_lookup import nested_lookup

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class WeightedMcc(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "weighted_mcc"
        self.moi = "Weighted Matthew Correlation Coefficient"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        if not "SVDD" in save_path:
            plt.xlabel("Learning Step")
            plt.ylabel("Accuracy")
            title = "Weighted vs Unweighted Matthew Correlation Coefficient"
            fig = plt.gcf()
            fig.suptitle(title, fontsize=16)

            ax = plt.gca()
            ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

            value_list = [i for sublist in nested_lookup(self.moi, logs["0-log-sample"]) for repeats in sublist for i in repeats]

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

            plt.legend(fontsize=7)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
            pdf.savefig()
            plt.close()
