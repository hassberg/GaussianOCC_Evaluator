import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_qs_name, get_model_name, get_dataset_name


class AverageBestLearningCurvePlotWithStd(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "average_of_best_learning_curves_with_std"
        self.moi = "Matthew Correlation Coefficient"
        self.k = 6

    def apply_metric(self, save_path: str, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.xlabel("Learning Step")
        plt.ylabel(self.moi)
        title = "Average Best Learning Curve"
        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path))

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
                    if iter_scoring[len(iter_scoring) - 1] > best_scoring:
                        best_scoring = iter_scoring[len(iter_scoring) - 1]
                        best = iter_scoring
                best_values.append(best)
            best_dict[model_qs] = best_values

        to_log = []
        for model_qs, scorings in best_dict.items():
            average_scoring = np.mean(scorings, axis=0)
            std = np.std(scorings, axis=0)
            plt.plot(range(len(average_scoring)), average_scoring, label=": ".join(model_qs.split("_")))
            plt.fill_between(range(len(average_scoring)), average_scoring - std, average_scoring + std, alpha=0.2)
            to_log.append([model_qs,
                           average_scoring[len(average_scoring) - 1],
                           std[len(std) - 1],
                           np.average([single[len(single) - 1] - single[len(single) - 1 - self.k] for single in scorings]),
                           np.average([single[0] for single in scorings]),
                           np.std([single[0] for single in scorings])]
                          )

        plt.legend(fontsize=5)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
        pdf.savefig()
        plt.close()

        with open(os.path.join(os.getcwd(), save_path, "best_run.txt"), "w+") as file:
            file.write("###############")
            file.write("\n")
            file.write(get_dataset_name(save_path))
            file.write("\n")
            for line in to_log:
                file.write("Model: " + line[0] + ", Mean: " + str(line[1]) + ", Std: " + str(line[2]) + ", Aeq: " + str(line[3]) + ", init: " + str(line[4]) + ", stdInit: " + str(line[5]))
                file.write("\n")
            file.write("###############")
