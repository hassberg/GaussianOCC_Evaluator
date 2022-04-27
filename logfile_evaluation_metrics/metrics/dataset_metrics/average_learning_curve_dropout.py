import os

from nested_lookup import nested_lookup

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_qs_name, get_model_name, get_dataset_name


class AverageLearningCurveDropout(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_dropout"
        self.moi = "Matthew Correlation Coefficient"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for dropout in self.dropout_boundaries:
            plt.xlabel("Learning Step")
            plt.ylabel(self.moi)
            plt.ylim(np.min(np.min(nested_lookup("Matthew Correlation Coefficient", logs))), 1)
            title = "Average Learning Curve with initial Quality geq " + str(dropout)

            fig = plt.gcf()
            fig.suptitle(title.replace("geq", ">"), fontsize=16)

            ax = plt.gca()
            ax.set_title(get_dataset_name(save_path))

            has_line = False
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
                        if iter_scoring[len(iter_scoring) - 1] - iter_scoring[0] > best_scoring:
                            best_scoring = iter_scoring[len(iter_scoring) - 1] - iter_scoring[0]
                            best = iter_scoring
                    best_values.append(best)
                best_dict[model_qs] = best_values

            for model_qs, curr_log in best_dict.items():
                # list contains best k runs:
                value_list = [k for i in nested_lookup(self.moi, curr_log) for subelem in i for k in subelem]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    has_line = True
                    average_scoring = np.mean(filtered_values, axis=0)
                    std = np.std(filtered_values, axis=0)
                    label = model_qs
                    plt.plot(range(len(average_scoring)), average_scoring, label=": ".join(model_qs.split("_")))
                    plt.fill_between(range(len(average_scoring)), average_scoring - std, average_scoring + std, alpha=0.2)
            if has_line:
                plt.legend(fontsize=4)
                if save_fig:
                    plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
                pdf.savefig()
            plt.close()
