from typing import List

from nested_lookup import nested_lookup, get_all_keys

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor

hyperparameter_list = ["LengthscaleLogger", "NoiseLogger", "VanishingLogger"]
hyperparameter_values = ["gamma", "lengthscale"]  # TODO kernel, C?


class HyperparameterSelected(EvalMetricExtractor):

    def __init__(self):
        self.name = "Hyperparameter"

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        list_params = {}
        for key in hyperparameter_list:
            if key in get_all_keys(dictonary):
                list_params[key] = nested_lookup(key, dictonary)

        value_params = {}
        for key in hyperparameter_values:
            if key in get_all_keys(dictonary):
                value_params[key] = nested_lookup(key, dictonary)
        return [list_params, value_params]
