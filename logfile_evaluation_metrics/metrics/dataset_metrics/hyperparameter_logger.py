import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class HyperparametreLogger(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "logged_hyperparameter"
        self.moi = "Hyperparameter"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):

        loggs = []
        all_lines = [(get_model_name(model) + "_" + get_qs_name(qs), logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            runs = []
            for run, dic in curr_log.items():
                for _, ls in dic.items():
                    runs.append([run.split("-")[0], ls[0][1]])
            loggs.append([model_qs, runs])

        with open(os.path.join(os.getcwd(), save_path, "hyper_params.txt"), "w+") as file:
            file.write("###############")
            file.write("\n")
            file.write(get_dataset_name(save_path))
            file.write("\n")
            for log in loggs:
                file.write("Model: " + log[0])
                file.write("\n")
                for run in log[1]:
                    stri = "run: " + run[0]
                    for parameter, val in run[1].items():
                        stri += " param: " + parameter + " ,val: " + str(val[0][0])
                    file.write(stri)
                    file.write("\n")
            file.write("###############")
