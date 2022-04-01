from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class AverageLearningCurvePlot(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iterations")
            plt.ylabel(metrics_name)
            title = "Average Learning Curve with initial scoring > " + str(dropout)
            plt.title(title)

            for key, values in logs.items():
                value_list = [i for subelems in values.items() for i in subelems[1]]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.mean(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + str(key).split("_")[0] + '\n' + \
                            str(key).split("_")[1]
                else:
                    average_scoring = np.mean(value_list, axis=0)
                    label = "**ALL**" + str(key).split("_")[0] + '\n' + str(key).split("_")[1]
                plt.plot(range(len(average_scoring)), average_scoring, label=label)

            plt.legend(fontsize=4)

            # plt.savefig("Average_learning_curve-dropout-" + str(dropout) + ".svg")
            pdf.savefig()
            plt.close()

        #printing standard deviation
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iterations")
            plt.ylabel(metrics_name)
            title = "Standard deviation of average Learning Curve with initial scoring > " + str(dropout)
            plt.title(title)

            for key, values in logs.items():
                value_list = [i for subelems in values.items() for i in subelems[1]]
                filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.std(filtered_values, axis=0)
                    label = "**" + str(len(filtered_values)) + "**" + str(key).split("_")[0] + '\n' + \
                            str(key).split("_")[1]
                else:
                    average_scoring = np.std(value_list, axis=0)
                    label = "**ALL**" + str(key).split("_")[0] + '\n' + str(key).split("_")[1]
                plt.plot(range(len(average_scoring)), average_scoring, label=label)

            plt.legend(fontsize=4)

            # plt.savefig("std_learning_curve-dropout-" + str(dropout) + ".svg")
            pdf.savefig()
            plt.close()
