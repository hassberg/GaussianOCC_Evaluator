from logfile_evaluation_metric import LogfileEvaluationMetric
from matplotlib.backends.backend_pdf import PdfPages
from nested_lookup import nested_lookup
import matplotlib.pyplot as plt


class QualityRangeMeasure(LogfileEvaluationMetric):
    def __init__(self):
        self.title = "Quality Range"

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
                    data.append(value[0][len(value[0]) - 1] - value[0][0])

            all_data.append(data)
            labels.append(log_name)

        fig1, ax = plt.subplots()
        ax.set_title(self.title)
        ax.boxplot(all_data, labels=labels)
        pdf.savefig(fig1)
