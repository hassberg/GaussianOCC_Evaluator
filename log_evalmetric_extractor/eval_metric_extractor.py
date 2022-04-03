from typing import List

import tensorflow as tf


class EvalMetricExtractor:
    def __init__(self, selection_option):
        self.name = None
        self.selection_option = selection_option

    def get_metrics_log(self, dictonary: dict) -> List[tf.Tensor]:
        pass

    def get_metric_as_dict(self, dict):
        metric_log = self.get_metrics_log(dict)
        return (self.name, metric_log)
