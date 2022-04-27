from typing import List

from nested_lookup import nested_lookup

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


class MatthewCorrelationCoefficientExtractor(EvalMetricExtractor):

    def __init__(self):
        self.name = "Matthew Correlation Coefficient"
        self.best_only = False

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        return nested_lookup("MccEval", dictonary)
