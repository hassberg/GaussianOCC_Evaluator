from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class LearningCurveOutlierSampled(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_per_outlier"
        self.outlier_sampled = [0, 1, 5, 10]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for os in self.outlier_sampled:
            plt.xlabel("Iterations")
            plt.ylabel(metrics_name)
            title = "Average Learning Curve with outlier sampled > " + str(os)
            plt.title(title)

            for key, values in logs.items():
                samples_w_mcc = [(i[1], i[0]) for subelems in values.items() for i in subelems[1]]

                outlier_sampled_w_mcc = list(map(lambda x: (sum(x[0]), x[1]), list(map(lambda x: (list(map(lambda y: float(y) * -0.5 + 0.5, x[0])), x[1]), samples_w_mcc))))

                filtered_values = list(map(lambda y: y[1], list(filter(lambda x: x[0] >= os, outlier_sampled_w_mcc))))

                if len(filtered_values) > 0:
                    average_scoring = np.mean(filtered_values, axis=0)
                    label = str(len(filtered_values)) + "_" + str(key).split("_")[0] + str(key).split("_")[1]
                    plt.plot(range(len(average_scoring)), average_scoring, label=label)
                else:
                    label = str(key).split("_")[0] + str(key).split("_")[1]
                    plt.plot(range(len(samples_w_mcc[1][1])), np.zeros(len(samples_w_mcc[1][1])), label=label)

            plt.legend(fontsize=4)

            plt.savefig("Average_learning_curve-dropout-" + str(os) + ".svg")
            pdf.savefig()
            plt.close()

            # printing standard deviation
            plt.xlabel("Iterations")
            plt.ylabel("Standard deviation of " + metrics_name)
            title = "Standard deviation of average Learning Curve with outlier sampled > " + str(os)
            plt.title(title)

            for key, values in logs.items():
                samples_w_mcc = [(i[1], i[0]) for subelems in values.items() for i in subelems[1]]

                outlier_sampled_w_mcc = list(map(lambda x: (sum(x[0]), x[1]), list(map(lambda x: (list(map(lambda y: float(y) * -0.5 + 0.5, x[0])), x[1]), samples_w_mcc))))

                filtered_values = list(map(lambda y: y[1], list(filter(lambda x: x[0] >= os, outlier_sampled_w_mcc))))

                if len(filtered_values) > 0:
                    average_scoring = np.std(filtered_values, axis=0)
                    label = str(key).split("_")[0] + str(key).split("_")[1]
                    plt.plot(range(len(average_scoring)), average_scoring, label=label)

            plt.legend(fontsize=4)

            plt.savefig("Std_average_learning_curve-dropout-" + str(os) + ".svg")
            pdf.savefig()
            plt.close()
