import os

from nested_lookup import nested_lookup

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


def get_color(label):
    if "-1.0 <=" in label:
        return 'r'
    elif "0.0 <=" in label:
        return 'g'
    elif "0.2 <=" in label:
        return 'b'
    elif "0.4 <=" in label:
        return 'y'
    else:
        raise RuntimeError


class AverageLearningCurveByDropoutRangeWithStd(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "average_learning_curve_by_dropout_range_with_error_bar"
        self.moi = "Matthew Correlation Coefficient"
        self.dropout_boundaries = [-1.0, 0.0, 0.2, 0.4, 1.0]

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for key, value in logs.items():
            plt.xlabel("Learning Step")
            plt.ylabel(self.moi)
            if key.split("-")[0] == "0":
                title = "Average Learning curve by initial correctness"
            else:
                title = "Average Learning curve by initial correctness of " + str(int(key.split("-")[0]) + 1)
            fig = plt.gcf()
            fig.suptitle(title, fontsize=16)

            ax = plt.gca()
            ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

            value_list = [i for sublist in nested_lookup(self.moi, value) for repeats in sublist for i in repeats]

            for i in range(len(self.dropout_boundaries) - 1):
                filtered_values = list(filter(lambda x: self.dropout_boundaries[i] <= x[0] < self.dropout_boundaries[i + 1], value_list))
                if len(filtered_values) >= 1:
                    average_scoring = np.mean(filtered_values, axis=0)
                    std = np.std(filtered_values, axis=0)
                    label = "Dropout: " + str(self.dropout_boundaries[i]) + " <= x < " + str(self.dropout_boundaries[i + 1])
                    plt.plot(range(len(average_scoring)), average_scoring, get_color(label),label=label)
                    plt.fill_between(range(len(average_scoring)), average_scoring - std, average_scoring + std, color=get_color(label), alpha=0.2)
            plt.legend(fontsize=4)
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
            pdf.savefig()
            plt.close()
