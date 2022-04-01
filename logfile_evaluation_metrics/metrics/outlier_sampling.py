from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


class OutlierSampling(LogfileEvaluationMetric):
    def __init__(self, ):
        self.dropout_boundaries = [-1.0, 0.1, 0.2, 0.3, 0.4]
        self.name = "outlier_sampled"

    def apply_metric(self, metrics_name: str, logs: dict, pdf: PdfPages):
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iteration")
            plt.ylabel("Average outlier")
            for key, values in logs.items():

                filtered = list(filter( lambda x: x[0][0] > dropout, [i for subelems in values.items() for i in subelems[1]]))
                if len(filtered) > 0:
                    samples = [i[1] for i in filtered]
                    label = str(len(filtered)) + "_" + key
                else:
                    samples = [i[1] for subelems in values.items() for i in subelems[1]]
                    label = "all_" + key

                outlier_sampled = list(map(lambda x: list(map(lambda y: float(y) * -0.5 + 0.5, x)), samples))
                avg_outlier = np.average(outlier_sampled, axis=0)
                avg_sampled_outlier = list(map(lambda x: sum(avg_outlier[:x]), range(len(avg_outlier) + 1)))

                plt.plot(range(len(avg_sampled_outlier)), avg_sampled_outlier, label=label)

            plt.legend(fontsize=4)
            plt.title("Outlier sampled. Dropout >" + str(dropout))
            # plt.savefig("Average_Outlier_Sampled-" + str(dropout) + ".svg")
            pdf.savefig()
            plt.close()

        #std..
        for dropout in self.dropout_boundaries:
            plt.xlabel("Iteration")
            plt.ylabel("Standard deviation of average outlier")
            for key, values in logs.items():

                filtered = list(filter( lambda x: x[0][0] > dropout, [i for subelems in values.items() for i in subelems[1]]))
                if len(filtered) > 0:
                    samples = [i[1] for i in filtered]
                    label = str(len(filtered)) + "_" + key
                else:
                    samples = [i[1] for subelems in values.items() for i in subelems[1]]
                    label = "all_" + key

                outlier_sampled = list(map(lambda x: list(map(lambda y: float(y) * -0.5 + 0.5, x)), samples))
                avg_outlier = np.std(outlier_sampled, axis=0)
                avg_sampled_outlier = list(map(lambda x: sum(avg_outlier[:x]), range(len(avg_outlier) + 1)))

                plt.plot(range(len(avg_sampled_outlier)), avg_sampled_outlier, label=label)

            plt.legend(fontsize=4)
            plt.title("Outlier sampled. Dropout >" + str(dropout))
            # plt.savefig("std_average_Outlier_Sampled-" + str(dropout) + ".svg")
            pdf.savefig()
            plt.close()
