import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class LearningCurveVsSample(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "learning_curve_vs_sampled"
        self.moi = "Mcc vs Sample"

    def apply_metric(self, save_path: str, logs: dict, pdf: PdfPages, save_fig: bool = False):
        fig1, ax = plt.subplots()
        plt.xlabel("Learning Step")
        plt.autoscale()
        ax.set_ylabel("Matthew Correlation Coefficient")
        title = "Sampled Label vs Matthew Correlation Coefficient"
        fig1.suptitle(title, fontsize=16)

        ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

        ax2 = ax.twinx()
        ax2.set_ylabel("Average Sampled Label")

        best_values = []
        dict_log = {}
        for sample, log in [(i, param_run[i]) for param_run in nested_lookup(self.moi, logs) for i in range(len(param_run))]:
            if sample in dict_log.keys():
                dict_log[sample].extend(log)
            else:
                dict_log[sample] = log
        for sample_name, sample_scoring in dict_log.items():
            best_scoring = -2
            best = None
            for iter_scoring in sample_scoring:
                if iter_scoring[0][len(iter_scoring[0]) - 1] > best_scoring:
                    best_scoring = iter_scoring[0][len(iter_scoring[0]) - 1]
                    best = iter_scoring
            best_values.append(best)


        avg_mcc = np.average([i[0] for i in best_values], axis=0)
        std_mcc = np.std([i[0] for i in best_values], axis=0)
        ax.plot(range(len(avg_mcc)), avg_mcc, label="mcc")
        ax.fill_between(range(len(avg_mcc)), avg_mcc + std_mcc, avg_mcc - std_mcc, alpha=0.2)

        avg_sample = np.average([i[1] for i in best_values], axis=0)
        ax2.plot([i + 1 for i in range(len(avg_sample))], avg_sample, '--', label="sample")

        pdf.savefig(fig1)
        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))

        plt.close(fig1)
