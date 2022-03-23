from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class AverageBestLearningCurvePlot(LogfileEvaluationMetric):
    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        plt.xlabel("Iterations")
        plt.ylabel(metrics_name)
        title = "Average Best Learning Curve  per Datasample "
        plt.title(title)

        for key, values in logs.items():
            best_values = []
            best = []
            best_scoring = -2
            for sample_name, sample_scoring in values.items():
                for iter_scoring in sample_scoring:
                    if iter_scoring[len(iter_scoring) - 1] - iter_scoring[0] > best_scoring:
                        best_scoring = iter_scoring[len(iter_scoring) - 1] - iter_scoring[0]
                        best = iter_scoring
                best_values.append(best)
                best_scoring = -2

            average_scoring = np.mean(best_values, axis=0)
            label = str(key).split("_")[0] + '\n' + str(key).split("_")[1]
            plt.plot(range(len(average_scoring)), average_scoring, label=label)

        plt.legend(fontsize=5)

        # plt.savefig("Average_lerning_curve_dropout-" + str(dropout) + ".svg")
        pdf.savefig()
        plt.close()
