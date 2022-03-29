from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from nested_lookup import nested_lookup


class CustomParameter(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "custom_progress_curves"
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:
            for key, values in logs.items():

                filtered_values = list(filter(lambda x: x["MccEval"][0] >= dropout, [i for i in nested_lookup("evaluations", values)]))

                parameter = ["lengthscale"]
                all_data = []
                for param in parameter:
                    all_data.append([i[0] for i in nested_lookup(param, filtered_values)])

                fig1, ax = plt.subplots()
                ax.set_yscale('log')
                ax.set_title(key + "\n dropout:" + str(dropout) + " samples:" + str(len(filtered_values)))
                ax.boxplot(all_data, labels=parameter)
                pdf.savefig(fig1)
                # plt.savefig("Average_lerning_curve_dropout-" + str(dropout) + ".svg")
                pdf.savefig()
                plt.close()
