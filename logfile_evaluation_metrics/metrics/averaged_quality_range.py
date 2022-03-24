from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class AverageQualityRange(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_quality_range"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]
        self.range_lengths = [3, 5, 10]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:
            for qr_length in self.range_lengths:
                plt.xlabel("Iterations")
                plt.ylabel(metrics_name + " improvement")
                title = "Average Quality Range with initial scoring >= " + str(dropout) + " and Range = " + str(qr_length)
                plt.title(title)
                plt.axhline(0, color='gray')
                plt.gca().set_ylim([-0.5,0.5])

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

                    quality_range = []
                    for i in range(len(average_scoring) - qr_length):
                        quality_range.append(average_scoring[i + qr_length] - average_scoring[i])

                    plt.plot(range(len(quality_range)), quality_range, label=label)

                plt.legend(fontsize=5)

                # plt.savefig("Average_quality_range_dropout-" + str(dropout) + "_range-" + str(qr_length) + ".svg")
                pdf.savefig()
                plt.close()
