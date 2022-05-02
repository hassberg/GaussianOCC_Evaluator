import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class WeightedLogger(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "logged_distinctiveness"
        self.moi = "Weighted Matthew Correlation Coefficient"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):

        all_lines = [(get_model_name(model) + "_" + get_qs_name(qs), logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        best_dict = {}
        for model_qs, curr_log in all_lines:
            best_values = []
            dict_log = {}
            for sample, log in [(i, param_run[i]) for param_run in nested_lookup(self.moi, curr_log) for i in range(len(param_run))]:
                if sample in dict_log.keys():
                    dict_log[sample].extend(log)
                else:
                    dict_log[sample] = log
            for sample_name, sample_scoring in dict_log.items():
                best_scoring = -2
                best = None
                for iter_scoring in sample_scoring:
                    if iter_scoring[len(iter_scoring) - 1][1] > best_scoring:
                        best_scoring = iter_scoring[len(iter_scoring) - 1][1]
                        best = iter_scoring
                best_values.append(best)
            best_dict[model_qs] = best_values

        loggs = []
        for model_qs, log in best_dict.items():
            runs = []
            if "MeanPrior" in model_qs:
                loggs.append([model_qs, [i[len(i)-1] for i in log]])

        with open(os.path.join(os.getcwd(), save_path, "weighted_log.txt"), "w+") as file:
            file.write("###############")
            file.write("\n")
            file.write(get_dataset_name(save_path))
            file.write("\n")
            for log in loggs:
                file.write("Model: " + log[0])
                avg = np.average(log[1], axis=0)
                std = np.std(log[1], axis=0)
                file.write(" weighted: " + str(avg[0]) + " std: " + str(std[0])  + " unweighted " + str(avg[1]) + " std: " + str(std[1]))
                file.write("\n")
            file.write("###############")
