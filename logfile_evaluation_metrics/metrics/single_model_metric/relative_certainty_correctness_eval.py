import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class RelativeCertaintyCorrectnessEval(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "relative_certainty_vs_misclassification"
        self.moi = "Relative Certainty vs Misclassification"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        if "SVDD" in save_path:
            return
        plt.xlabel("Learning Step")
        plt.ylabel("Misclassification ratio")
        title = "Misclassification by prediction distinctiveness"
        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

        value_list = [i for sublist in nested_lookup(self.moi, logs["0-log-sample"]) for repeats in sublist for i in repeats]

        mis_rate = []
        for i in range(len(value_list)):
            if mis_rate == []:
                lst = value_list[i]
                mis_rate = [[i[0]] for i in lst]
            else:
                for j in range(len(value_list[i])):
                    mis_rate[j].append(value_list[i][j][0])

        mean = np.average(mis_rate, axis=1)
        std = np.std(mis_rate, axis=1)
        x = (0.5, 0.65, 0.75, 0.85, 1.0)

        x = (1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.5)
        mean = np.asarray([mean[0], mean[0], mean[1], mean[2], mean[3], mean[4], mean[4]])
        std = np.asarray([std[0], std[0], std[1], std[2], std[3], std[4], std[4]])

        plt.plot(x, mean)
        plt.fill_between(x, mean + std, mean - std, alpha=0.2)


        # if save_fig:
        plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        # pdf.savefig()

        plt.close()
