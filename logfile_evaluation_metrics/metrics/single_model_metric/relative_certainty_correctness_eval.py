import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class RelativeCertaintyCorrectnessEval(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "relative_certainty_vs_misclassification"
        self.moi = "Relative Certainty vs Misclassification"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.xlabel("Iterations")
        plt.ylabel(self.moi)
        title = "Relative Certainty vs. Misclassification"
        plt.title(title)
        plt.ylim(0, 1)

        value_list = [i for sublist in nested_lookup(self.moi, logs) for repeats in sublist for i in repeats]

        itterations = []
        for i in range(len(value_list)):
            if itterations == []:
                itterations = [[i] for i in value_list[0]]
            else:
                for j in range(len(value_list[i])):
                    itterations[j].append(value_list[i][j])

        labels = ["90", "80", "70", "60", "50"]
        followup = "_percent"
        for i in range(len(labels)):
            steping = []
            for step in range(len(itterations)):
                steping.append([curr[i] for curr in itterations[step]])
            mean = np.average(steping, axis=1)
            std = np.std(steping, axis=1)
            plt.plot(range(1, len(steping) + 1), mean, label=str(labels[i]) + followup)
            plt.fill_between(range(1, len(steping) + 1), mean + std, mean - std, alpha=0.3)

        plt.legend(fontsize=4)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
