from logfile_evaluation_metrics.logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt
import numpy as np


class AverageEndQualityMeasure(LogfileEvaluationMetric):
    def __init__(self):
        self.title = "Average end Quality"
        self.k = 10

    def apply(self, logs: [dict], pdf: PdfPages):
        all_data = []
        labels = []
        for log_name, log_result in logs:
            data = []
            for single_logging_result in log_result:
                # TODO here apply metric..
                found_statistics = nested_lookup("BasicActiveLearningCurveMetric", single_logging_result,
                                                 with_keys=True)
                for value in found_statistics.values():
                    end_quality_sum = 0.0
                    for i in range(len(value[0]) - self.k, len(value[0])):
                        end_quality_sum = end_quality_sum + value[0][i]
                    data.append(np.multiply(np.divide(1, self.k), end_quality_sum))

            all_data.append(data)
            labels.append(log_name)

        fig1, ax = plt.subplots()
        ax.set_title(self.title)
        ax.boxplot(all_data, labels=labels)
        pdf.savefig(fig1)
