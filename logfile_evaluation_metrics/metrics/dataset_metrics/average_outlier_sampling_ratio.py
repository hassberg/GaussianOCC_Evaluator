import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial.distance import pdist
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from name_wrapper import get_dataset_name, get_model_name, get_qs_name


class AverageOutlierSamplingRatio(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "average_outlier_sampling_ratio"
        self.moi = "Sampled Labels"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.ylabel("Outlier sampling ratio")
        title = "Average Outlier Sampling Ratio"
        plt.autoscale()

        fig = plt.gcf()
        fig.suptitle(title, fontsize = 16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path))

        average_ratio = []

        all_lines = [(get_model_name(model) + "_" + get_qs_name(qs), logs[model][qs]) for model in logs.keys() for qs in logs[model]]
        for model_qs, curr_log in all_lines:
            # list contains best k runs:
            value_list = [k for i in nested_lookup(self.moi, curr_log["0-log-sample"]) for subelem in i for k in subelem]
            average_ratio.append((model_qs, np.sum((np.asarray(value_list) * -1 +1 )*0.5, axis=1)))

        plt.boxplot([i[1] for i in average_ratio], labels=[i[0].replace("_", "\n") for i in average_ratio])
        plt.xticks(fontsize=5, rotation=17)

        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".pdf"))
        pdf.savefig()
        plt.close()

        to_log = []
        for model in average_ratio:
            to_log.append([model[0], np.average(model[1]), np.std(model[1])])

        with open(os.path.join(os.getcwd(), save_path, "os_rate.txt"), "w+") as file:
            file.write("###############")
            file.write("\n")
            file.write(get_dataset_name(save_path))
            file.write("\n")
            for line in to_log:
                file.write("Model: " + line[0] + ", Mean: " + str(line[1]) +", Std: " + str(line[2]))
                file.write("\n")
            file.write("###############")
