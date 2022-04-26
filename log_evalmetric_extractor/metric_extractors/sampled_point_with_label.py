from typing import List

from nested_lookup import nested_lookup

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


class SampledPointsWithLabel(EvalMetricExtractor):

    def __init__(self):
        self.name = "Sampled Points With Label"

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        points = list(map(lambda x: [i for points in x for i in points], [nested_lookup("actual_queries", i) for i in dictonary.values()]))
        labels = list(map(lambda x: [i for points in x for k in points for i in k], [nested_lookup("query_results", i) for i in dictonary.values()]))
        res = []
        for i in len(points):
            res.append([pt for pt in zip(labels[i],points[i])])
        return res

