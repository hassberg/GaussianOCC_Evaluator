from logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup

import matplotlib.pyplot as plt
import numpy as np


class OutlierSamplingMeasure(LogfileEvaluationMetric):
    def __init__(self):
        self.title = "Outlier Sampled"

    def apply(self, logs: [dict], pdf: PdfPages):
        all_data = []
        labels = []
        for log_name, log_result in logs:
            data = []
            for single_logging_result in log_result:
                # TODO here apply metric..
                sampled_scores = sum(
                    [item for sublist in nested_lookup("query_results", single_logging_result) for item in sublist], [])
                data.append(np.divide(sampled_scores.count(-1), len(sampled_scores)))

            all_data.append(data)
            labels.append(log_name)

        fig1, ax = plt.subplots()
        ax.set_title(self.title)
        ax.boxplot(all_data, labels=labels)
        pdf.savefig(fig1)
