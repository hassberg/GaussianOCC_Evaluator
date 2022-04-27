from typing import List

from nested_lookup import nested_lookup

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


class SampledLabel(EvalMetricExtractor):

    def __init__(self):
        self.name = "Sampled Labels"
        self.best_only = True

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        return list(map(lambda x: [i for points in x for k in points for i in k], [nested_lookup("query_results", i) for i in dictonary.values()]))
