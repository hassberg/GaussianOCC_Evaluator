import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_qs_name, get_model_name


def get_mean(list):
    stepwise = []
    for iteration in range(len(list)):
        if len(list[iteration]) == 0:
            stepwise.append(0)
        else:
            stepwise.append(np.mean(list[iteration]))
    return np.asarray(stepwise)


def get_std(list):
    stepwise = []
    for iteration in range(len(list)):
        if len(list[iteration]) == 0:
            stepwise.append(0)
        else:
            stepwise.append(np.mean(list[iteration]))
    return np.asarray(stepwise)


class UncertaintyConfusionDev(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "uncertainty_confusion_correlation"
        self.moi = "Uncertainty Confusion Correlation"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.xlabel("Learning Step")
        plt.ylabel("Uncertainty")
        title = "Uncertainty per Prediction case"
        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=9)

        value_list = [i for sublist in nested_lookup(self.moi, logs) for repeats in sublist for i in repeats]

        iter = []
        for run in value_list:
            if iter == []:
                iter = [[i] for i in run]

            else:
                for step in range(len(run)):
                    iter[step].append(run[step])

        labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
        for field in range(len(labels)):
            steping = []
            for i in range(len(iter)):
                steping.append([single for step in iter[i] for single in step[field]])

            mean = get_mean(steping)
            std = get_std(steping)
            plt.plot(range(len(mean)), mean, label=labels[field])
            plt.fill_between(range(len(mean)), mean + std, mean - std, alpha=0.3)

        plt.legend(fontsize=4)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
