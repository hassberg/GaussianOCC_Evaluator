import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class AverageLearningCurvePlot(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "average_learning_curve"
        self.metric_of_interest = "Matthew Correlation Coefficient"
        self.dropout_boundaries = [-1.0, 0.1, 0.3]

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iterations")
            plt.ylabel(self.metric_of_interest)
            title = "Average Learning Curve with initial scoring geq " + str(dropout)
            plt.title(title.replace("geq", ">"))

            all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
            for model_qs, curr_log in all_lines:
                # list contains best k runs:
                value_list = [k for i in nested_lookup(self.metric_of_interest, curr_log) for subelem in i for k in subelem]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.mean(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + model_qs
                else:
                    average_scoring = np.mean(value_list, axis=0)
                    label = "**ALL**" + model_qs
                plt.plot(range(len(average_scoring)), average_scoring, label=label)

            plt.legend(fontsize=4)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            pdf.savefig()
            plt.close()

            # printing standard deviation
            plt.xlabel("Iterations")
            plt.ylabel(self.metric_of_interest)
            title = "Standard deviation of average learning curve with initial scoring geq " + str(dropout)
            plt.title(title.replace("geq", ">"))

            all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
            for model_qs, curr_log in all_lines:
                # list contains best k runs:
                value_list = [k for i in nested_lookup(self.metric_of_interest, curr_log) for subelem in i for k in subelem]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.std(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + model_qs
                else:
                    average_scoring = np.std(value_list, axis=0)
                    label = "**ALL**" + model_qs
                plt.plot(range(len(average_scoring)), average_scoring, label=label)

            plt.legend(fontsize=4)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            pdf.savefig()
            plt.close()
