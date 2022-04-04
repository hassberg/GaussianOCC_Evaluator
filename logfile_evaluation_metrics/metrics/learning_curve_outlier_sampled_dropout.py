import os

from nested_lookup import nested_lookup

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


## uses baseline as ... instead of initial correctness
class AverageLearningCurveDropoutOutlierSampled(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_dropout_by_outlier"
        self.outlier_sampled = [0, 1, 5, 10]
        self.metric_of_interest = "Mcc vs Sample"

    def apply_metric(self, save_path: str, logs: dict, pdf: PdfPages, save_fig: bool = False):
        logs_with_outlier_sampled = []
        all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            # list contains best k runs:
            value_list = [k for i in nested_lookup(self.metric_of_interest, curr_log) for subelem in i for k in subelem]
            outlier_sampled_w_mcc = list(map(lambda x: (sum(x[0]), x[1]), list(map(lambda x: (list(map(lambda y: float(y) * -0.5 + 0.5, x[1])), x[0]), value_list))))
            logs_with_outlier_sampled.append((model_qs, outlier_sampled_w_mcc))

        for oss in self.outlier_sampled:
            plt.xlabel("Iterations")
            plt.ylabel("Mcc")
            title = "Average Learning Curve with outlier sampled gt " + str(oss)
            for model_qs, log_with_os in logs_with_outlier_sampled:
                plt.title(title.replace("gt", ">"))

                filtered_values = list(map(lambda y: y[1], list(filter(lambda x: x[0] >= oss, log_with_os))))
                if len(filtered_values) > 0:
                    average_scoring = np.mean(filtered_values, axis=0)
                    label = str(len(filtered_values)) + "_" + model_qs
                    plt.plot(range(len(average_scoring)), average_scoring, label=label)
                else:
                    label = "**all**_" + model_qs
                    plt.plot(range(len(log_with_os[1][1])), np.zeros(len(log_with_os[1][1])), label=label)

            plt.legend(fontsize=4)

            pdf.savefig()
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close()

            ## std..
            plt.xlabel("Iterations")
            plt.ylabel("Mcc")
            title = "Standard deviation of average learning curve with outlier sampled gt " + str(oss)
            for model_qs, log_with_os in logs_with_outlier_sampled:
                plt.title(title.replace("gt", ">"))

                filtered_values = list(map(lambda y: y[1], list(filter(lambda x: x[0] >= oss, log_with_os))))
                if len(filtered_values) > 0:
                    average_scoring = np.std(filtered_values, axis=0)
                    label = str(len(filtered_values)) + "_" + model_qs
                    plt.plot(range(len(average_scoring)), average_scoring, label=label)
                else:
                    label = "**all**_" + model_qs
                    plt.plot(range(len(log_with_os[1][1])), np.zeros(len(log_with_os[1][1])), label=label)

            plt.legend(fontsize=4)

            pdf.savefig()
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close()
