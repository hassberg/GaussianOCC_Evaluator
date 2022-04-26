import os

import numpy as np

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from nested_lookup import nested_lookup, get_all_keys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from name_wrapper import get_dataset_name, get_qs_name, get_model_name


class HyperparameterEvalMetric(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "hyperparameter"
        self.moi = "Hyperparameter"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        # compare learning of k best
        list_params = []
        value_params = []
        array_params = []
        for lst in nested_lookup("Hyperparameter", logs):
            list_params.append([x[0] for x in lst])
            value_params.append([x[1] for x in lst])
            array_params.append([x[2] for x in lst])


        ## plot learning..
        if len(get_all_keys(list_params)) > 0:
            plt.xlabel("Learning Step")
            plt.ylabel(self.moi)
            title = "Learning curve of Parameters.."
            fig = plt.gcf()
            fig.suptitle(title, fontsize = 16)

            ax = plt.gca()
            ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True))

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
            fig1, ax = plt.subplots(1, len(params))
            title = "Grid searched best Parameters.."
            fig1.suptitle(title)
            if not isinstance(ax, np.ndarray):
                ax = [ax]

            for i in range(len(params)):
                ax[i].set_ylabel(params[i][0])
                ax[i].boxplot([[i[0] for i in params[i][1]]], labels=[params[i][0]])

            pdf.savefig(fig1)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close(fig1)

        # rd..
        params = []
        for key in get_all_keys(array_params[0][0]):
            params.append((key, [x[0] for i in range(len(array_params)) for x in nested_lookup(key, array_params[i][0])]))
        for key, values in params:
            title = "Learning Curves of Parameter.."+ key
            plt.xlabel("Iterations")
            plt.ylabel(key)
            plt.title(title)
            for value in values:
                plt.plot(range(len(value)), value)

            pdf.savefig()
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close()
