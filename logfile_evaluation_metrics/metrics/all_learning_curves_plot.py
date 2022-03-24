from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class AllLearningCurvesPlot(LogfileEvaluationMetric):
    def __init__(self):
        self.name = "all_learning_curves"

    def apply_metric(self,metrics_name,  logs: dict, pdf: PdfPages):
        for key, values in logs.items():

            plt.xlabel("Iterations")
            plt.ylabel(metrics_name)
            title = str(key).split("_")[0] + '\n' + str(key).split("_")[1]
            plt.title(title)

            for single_run_scoring in [i for subelems in values.items() for i in subelems[1]]:
                plt.plot(range(len(single_run_scoring)), single_run_scoring)

            pdf.savefig()
            plt.close()
