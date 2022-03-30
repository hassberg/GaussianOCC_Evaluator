from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from nested_lookup import nested_lookup


class TrainableProgress(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "trainable_progress_curves"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:
            for key, values in logs.items():
                filtered_values = list(filter(lambda x: x["MccEval"][0] >= dropout, [i for i in nested_lookup("evaluations", values)]))

                plt.xlabel("Iterations")
                title = key + "\ndropout:" + str(dropout) + " samples:" + str(len(filtered_values))
                plt.title(title)
                plt.axhline(0, color='gray')
                ## mcc
                value_list = nested_lookup("MccEval", filtered_values)
                average_scoring = np.mean(value_list, axis=0)
                label = "Mcc Scoring on evaluation Data"
                plt.plot(range(len(average_scoring)), average_scoring, label=label)
                ## lengthscale
                value_list = nested_lookup("LengthscaleLogger", filtered_values)
                average_scoring = np.mean(value_list, axis=0)
                label = "Logged Lengthscale"
                plt.plot(range(len(average_scoring)), average_scoring, label=label)
                ## noise
                value_list = nested_lookup("NoiseLogger", filtered_values)
                average_scoring = np.mean(value_list, axis=0)
                label = "Logged Noise"
                plt.plot(range(len(average_scoring)), average_scoring, label=label)

                plt.legend(fontsize=4)

                # plt.savefig("Self_Training_parameter-dropout-" + str(dropout) + ".svg")
                pdf.savefig()
                plt.close()
