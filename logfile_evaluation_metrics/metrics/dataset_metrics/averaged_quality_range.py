import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class AverageQualityRange(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_quality_range"
        self.moi = "Matthew Correlation Coefficient"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]
        self.range_lengths = [3, 5, 10]

    def apply_metric(self, save_path: str, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for dropout in self.dropout_boundaries:
            for qr_length in self.range_lengths:
                plt.xlabel("Iterations")
                plt.ylabel(self.moi + " improvement")
                title = "Average Quality Range with initial scoring geq " + str(dropout) + " and Range = " + str(qr_length)
                plt.title(title.replace("geq", ">="))
                plt.axhline(0, color='gray')
                plt.gca().set_ylim([-0.5, 0.5])

                all_lines = [(model + "_" + qs, logs[model][qs]) for model in logs.keys() for qs in logs[model]]
                for model_qs, curr_log in all_lines:
                    # list contains best k runs:
                    value_list = [k for i in nested_lookup(self.moi, curr_log) for subelem in i for k in subelem]
                    filtered_values = list(filter(lambda x: x[0] >= dropout, value_list))

                    if len(filtered_values) >= 1:
                        average_scoring = np.mean(filtered_values, axis=0)
                        label = "**" + str(len(filtered_values)) + "**" + model_qs
                    else:
                        average_scoring = np.mean(value_list, axis=0)
                        label = "**ALL**" + model_qs

                    quality_range = []
                    for i in range(len(average_scoring) - qr_length):
                        quality_range.append(average_scoring[i + qr_length] - average_scoring[i])

                    plt.plot(range(len(quality_range)), quality_range, label=label)

                plt.legend(fontsize=4)
                if save_fig:
                    plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
                pdf.savefig()
                plt.close()
