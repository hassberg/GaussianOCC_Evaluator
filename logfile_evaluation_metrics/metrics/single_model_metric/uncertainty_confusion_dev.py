import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


def average_step_confusion_field(scoring):
    ## Aufteilung je step in tp fp fn tn
    avg = []
    for field in range(4):
        valid_points = []
        for i in range(len(scoring)):
            if isinstance(scoring[i][field], np.ndarray):
                valid_points.append(scoring[i][field])
        avg.append(np.average(valid_points, axis=0))

    return avg


def calc_certainty(mean, std):
    return np.divide(np.abs(mean), np.sqrt(std))


class UncertaintyConfusionDev(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "uncertainty_confusion_correlation"
        self.moi = "Uncertainty Confusion Correlation"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for key, value in logs.items():
            plt.xlabel("Iterations")
            plt.ylabel(self.moi)
            title = "Uncertainty-Confusion Correlation " + str(int(key.split("-")[0]) + 1)
            plt.title(title)

            value_list = [i for sublist in nested_lookup(self.moi, value) for repeats in sublist for i in repeats]

            iter = []
            for run in value_list:
                if iter == []:
                    iter = [[i] for i in run]

                else:
                    for step in range(len(run)):
                        iter[step].append(run[step])

            dev_curve = []
            for i in range(len(iter)):
                dev_curve.append(average_step_confusion_field(iter[i]))

            certainty_curve = []
            for i in range(len(dev_curve)):
                step = []
                for field in range(4):
                    step.append(calc_certainty(dev_curve[i][field][0], dev_curve[i][field][1]))  # TODO variance?
                certainty_curve.append(step)

            labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
            for i in range(len(labels)):
                plt.plot(range(1, len(certainty_curve) + 1), [pt[i] for pt in certainty_curve], label=labels[i])

            plt.legend(fontsize=4)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            pdf.savefig()
            plt.close()
