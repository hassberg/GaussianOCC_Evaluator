from typing import List

from nested_lookup import nested_lookup

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


class SampledPoints(EvalMetricExtractor):

    def __init__(self):
        self.name = "Sampled Points"

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        return list(map(lambda x: [i for points in x for i in points], [nested_lookup("actual_queries", i) for i in dictonary.values()]))
