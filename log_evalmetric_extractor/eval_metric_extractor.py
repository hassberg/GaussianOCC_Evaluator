from typing import List

import tensorflow as tf


class EvalMetricExtractor:
    def __init__(self, selection_option):
        self.name = None
        self.selection_option = selection_option
        self.best_only = False

    def get_metrics_log(self, dictonary: dict) -> List[tf.Tensor]:
        pass

    def get_metric_as_dict(self, dict, sample_name):
        if self.best_only and "0-log-sample" not in sample_name:
            return (self.name, [])
        metric_log = self.get_metrics_log(dict)
        return (self.name, metric_log)
