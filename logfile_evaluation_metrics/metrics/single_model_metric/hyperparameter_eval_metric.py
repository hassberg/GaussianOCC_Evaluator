import os

import numpy as np

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from nested_lookup import nested_lookup, get_all_keys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class HyperparameterEvalMetric(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "hyperparameter"
        self.moi = "Hyperparameter"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        # compare learning of k best
        list_params = []
        value_params = []
        for lst in nested_lookup("Hyperparameter", logs):
            list_params.append([x[0] for x in lst])
            value_params.append([x[1] for x in lst])

        ## plot learning..
        plt.xlabel("Iterations")
        plt.ylabel(self.moi)
        title = "Learning curve of Parameters.."
        plt.title(title)
        for i in range(len(list_params)):
            ith = list_params[i]
            for key in get_all_keys(ith[0]):
                mean = np.mean([m for x in ith for l in nested_lookup(key, x) for m in l], axis=0)
                plt.plot(range(len(mean)), mean, label=str(i + 1) + "-" + key)

        plt.legend(fontsize=5)
        pdf.savefig()
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        plt.close()

        # grafik k best abs.
        params = []
        for key in get_all_keys(value_params[0][0]):
            params.append((key, [x[0] for i in range(len(value_params)) for x in nested_lookup(key, value_params[i][0])]))
        if len(params) > 0:
            plt.xlabel("Iterations")
            plt.ylabel(self.moi)
            title = "Grid searched best Parameters.."
            plt.title(title)

            labels = []
            values = []
            for i in range(len(params)):
                labels.append(params[i][0])
                values.append([i[0] for i in params[i][1]])

            plt.boxplot(values, labels=labels)

            pdf.savefig()
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close()
