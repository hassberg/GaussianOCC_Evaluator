from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class AverageLearningCurveWithErrorBar(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_with_error_bar"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    # TODO impl this with a shaded variance..
    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:

            for key, values in logs.items():
                plt.xlabel("Iterations")
                plt.ylabel(metrics_name)
                title = "Average Learning Curve with initial scoring > " + str(dropout)
                plt.title(title)
                plt.gca().set_ylim([-1, 1])
                value_list = [i for subelems in values.items() for i in subelems[1]]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.mean(filtered_values, axis=0)
                    average_error = np.std(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + str(key).split("_")[0] + '\n' + \
                            str(key).split("_")[1]
                else:
                    average_scoring = np.mean(value_list, axis=0)
                    average_error = np.std(value_list, axis=0)
                    label = "**ALL**" + str(key).split("_")[0] + '\n' + str(key).split("_")[1]
                # plt.plot(range(len(average_scoring)), average_scoring, label=label)
                if average_scoring[len(average_scoring) -1] - average_scoring[0] > 0.01:
                    plt.errorbar(x=range(len(average_scoring)), y=average_scoring, yerr=average_error, label=label)

                    plt.legend(fontsize=4)

                    # plt.savefig("Average_lerning_curve_dropout-" + str(dropout) + ".svg")
                    pdf.savefig()
                plt.close()
