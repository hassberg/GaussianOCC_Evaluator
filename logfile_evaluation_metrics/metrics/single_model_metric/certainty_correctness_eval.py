import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_qs_name, get_model_name


class CertaintyCorrectnessEval(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "certainty_vs_correctness"
        self.moi = "Uncertainty vs Misclassification"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.xlabel("Learning Step", fontsize=16)
        plt.ylabel("Misclassification ratio", fontsize=16)
        title = "Misclassification Ratio by Relative Uncertainty"

        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

        plt.ylim(0, 0.99)

        value_list = [i for sublist in nested_lookup(self.moi, logs["0-log-sample"]) for repeats in sublist for i in repeats]

        itterations = []
        for i in range(len(value_list)):
            if itterations == []:
                lst = []
                for j in range(len(value_list[i])):
                    lst.append([p[1] for p in value_list[i][j]])
                itterations = [[i] for i in lst]
            else:
                for j in range(len(value_list[i])):
                    itterations[j].append([p[1] for p in value_list[i][j]])

        end_uncert = np.average([[i[0][0], i[1][0], i[2][0], i[3][0], i[4][0], ] for i in [i[len(value_list)-1] for i in value_list ]], axis = 0)
        followup = ". Certainty Group"
        for i in range(5):
            steping = []
            for step in range(len(itterations)):
                steping.append([curr[i] for curr in itterations[step]])
            mean = np.average(steping, axis=1)
            std = np.std(steping, axis=1)
            plt.plot(range(1, len(steping) + 1), mean, label=str(i+1) + followup + ", (" + str(end_uncert[i])[0:4] + ")")
            plt.fill_between(range(1, len(steping) + 1), mean + std, mean - std, alpha=0.2)

        plt.legend(fontsize=9)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
        pdf.savefig()
        plt.close()
