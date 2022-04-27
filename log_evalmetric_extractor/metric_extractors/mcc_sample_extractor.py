from typing import List, Tuple

from nested_lookup import nested_lookup

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor
import numpy as np


class MccSampleExtractor(EvalMetricExtractor):

    def __init__(self):
        self.name = "Mcc vs Sample"
        self.best_only = True

    def get_metrics_log(self, dictionary: dict) -> []:
        log = []
        for key, value in dictionary.items():
            sampled = [np.asarray(i).flatten()[0] for i in nested_lookup("query_results", value)]
            mcc = nested_lookup("MccEval", value)
            log.append((mcc[0], sampled))

        return log

