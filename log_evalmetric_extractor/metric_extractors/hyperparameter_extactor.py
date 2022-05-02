from typing import List

from nested_lookup import nested_lookup, get_all_keys

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor

hyperparameter_list = ["LengthscaleLogger"]
hyperparameter_values = ["gamma", "lengthscale", "C"]  # TODO kernel, C?
hyperparameter_array = ["VanishingLogger"]


class HyperparameterSelected(EvalMetricExtractor):

    def __init__(self):
        self.name = "Hyperparameter"
        self.best_only = False

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        list_params = {}
        for key in hyperparameter_list:
            if key in get_all_keys(dictonary):
                list_params[key] = nested_lookup(key, dictonary)

        value_params = {}
        for key in hyperparameter_values:
            if key in get_all_keys(dictonary):
                value_params[key] = nested_lookup(key, dictonary)

        array_params ={}
        for key in hyperparameter_array:
            if key in get_all_keys(dictonary):
                array_params[key] = nested_lookup(key, dictonary)
        return [list_params, value_params, array_params]
