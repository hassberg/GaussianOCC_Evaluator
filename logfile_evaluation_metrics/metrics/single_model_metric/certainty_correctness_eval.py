import os

from nested_lookup import nested_lookup
from scipy.ndimage import label

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from name_wrapper import get_dataset_name, get_qs_name, get_model_name


class CertaintyCorrectnessEval(LogfileEvaluationMetric):
    def __init__(self, ):
        self.name = "certainty_vs_correctness"
        self.moi = "Uncertainty vs Misclassification"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        plt.xlabel("Learning Step", fontsize=16)
        plt.ylabel("Misclassification ratio", fontsize=16)
        title = "Misclassification Ratio by Relative Uncertainty"

        fig = plt.gcf()
        fig.suptitle(title, fontsize=16)

        ax = plt.gca()
        ax.set_title(get_dataset_name(save_path) + ", " + get_model_name(save_path, True) + ", " + get_qs_name(save_path, True), fontsize=7)

        # plt.ylim(0, 0.99)

        value_list = [i for sublist in nested_lookup(self.moi, logs["0-log-sample"]) for repeats in sublist for i in repeats]

        groups = []
        mis_rate = []
        for i in range(len(value_list)):
            if groups == []:
                lst = value_list[i]
                groups = [[i[0]] for i in lst]
                mis_rate = [[i[1]] for i in lst]
            else:
                for j in range(len(value_list[i])):
                    groups[j].append(value_list[i][j][0])
                    mis_rate[j].append(value_list[i][j][1])




        gp_avg = np.average(groups, axis=1)
        gp_std = np.std(groups, axis=1)
        mis_rate_avg = np.average(mis_rate, axis=1)
        mis_rate_std = np.std(mis_rate, axis=1)


        # plt.boxplot(mis_rate, positions=gp_avg)
        plt.plot(gp_avg, mis_rate_avg)
        plt.fill_between(gp_avg, mis_rate_avg + mis_rate_std, mis_rate_avg - mis_rate_std, alpha=0.2)


        plt.xticks(gp_avg, [str(i)[0:3] for i in gp_avg])

        if save_fig:
            plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
        pdf.savefig()
        plt.close()
