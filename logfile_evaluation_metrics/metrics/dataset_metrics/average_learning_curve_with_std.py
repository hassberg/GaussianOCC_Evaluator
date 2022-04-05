import os

from nested_lookup import nested_lookup

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class AverageLearningCurveWithStd(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_with_error_bar"
        self.moi = "Matthew Correlation Coefficient"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iterations")
            plt.ylabel(self.moi)
            title = "Average learning curve and standard deviation with initial scoring geq " + str(dropout)
            plt.title(title.replace("geq", ">"))
            has_line = False
            all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
            for model_qs, curr_log in all_lines:
                # list contains best k runs:
                value_list = [k for i in nested_lookup(self.moi, curr_log) for subelem in i for k in subelem]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    has_line = True
                    average_scoring = np.mean(filtered_values, axis=0)
                    std = np.std(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + model_qs
                    plt.plot(range(len(average_scoring)), average_scoring, label=label)
                    plt.fill_between(range(len(average_scoring)), average_scoring - std, average_scoring + std, alpha=0.2)
            if has_line:
                plt.legend(fontsize=4)
                if save_fig:
                    plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
                pdf.savefig()
            plt.close()
