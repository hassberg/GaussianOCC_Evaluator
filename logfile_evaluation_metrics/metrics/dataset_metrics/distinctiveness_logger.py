import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


def get_averaged(log):
    score = [[], [], [], [], []]
    size = [[], [], [], [], []]
    for run in range(len(log[1])):
        for i in range(5):
            score[i].append(log[1][run][i][0])
            size[i].append(log[1][run][i][1])

    for i in range(5):
        score[i] = [np.average(score[i]), np.std(score[i])]
        size[i] = np.average(size[i])

    sum_size = np.sum(size)

    for i in range(5):
        size[i] = size[i] / sum_size

    return size, score


class DistinctivenessLogger(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "logged_distinctiveness"
        self.moi = "Relative Certainty vs Misclassification"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):

        loggs = []
        all_lines = [(get_model_name(model) + "_" + get_qs_name(qs), logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            runs = []
            for run in nested_lookup("0-log-sample", curr_log):
                loggs.append([model_qs, [i[len(i) - 1] for _, k in run.items() for j in k for i in j]])

        with open(os.path.join(os.getcwd(), save_path, "distinctiveness.txt"), "w+") as file:
            file.write("###############")
            file.write("\n")
            file.write(get_dataset_name(save_path))
            file.write("\n")
            for log in loggs:
                size, score = get_averaged(log)
                file.write("Model: " + log[0])
                file.write("\n")
                for i in range(5):
                    file.write(" _" + str(i) + " Mean: " + str(score[i][0]) + " Std: " + str(score[i][1]) + " Size: " + str(size[i]))
                    file.write("\n")
            file.write("###############")
