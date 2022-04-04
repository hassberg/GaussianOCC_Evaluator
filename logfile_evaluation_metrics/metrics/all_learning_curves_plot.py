import os

from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from nested_lookup import nested_lookup
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class AllLearningCurvesPlot(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "all_learning_curves"
        self.metric_of_interest = "Matthew Correlation Coefficient"

    def apply_metric(self, save_path, logs: dict, pdf: PdfPages, save_fig: bool = False):
        for key, values in logs.items():

            plt.xlabel("Iterations")
            plt.ylabel(self.metric_of_interest)
            title = "All learning curves of " + str(int(key) + 1) + ". best Parameter selection"
            plt.title(title)

            for single_run_scoring in [k for i in nested_lookup(self.metric_of_interest, values) for subelem in i for k in subelem]:
                plt.plot(range(len(single_run_scoring)), single_run_scoring)

            title.lower()
            pdf.savefig()
            if save_fig:
                plt.savefig(os.path.join(save_path, title.lower().replace(" ", "_") + ".svg"))
            plt.close()
