from typing import List
from sklearn.metrics import matthews_corrcoef
from nested_lookup import nested_lookup, get_all_keys
import numpy as np

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


def get_certainty_share(mean, std):
    lower = mean - 2 * std
    upper = mean + 2 * std

    if lower > 0 or upper < 0:
        return 1.0
    else:
        rge = np.abs(lower) + np.abs(upper)
        return np.maximum(np.abs(lower), np.abs(upper)) / rge


def get_gp_prediction(mean) -> int:
    if (mean >= 0):
        return 1
    else:
        return -1


def weighted_accuracy(uncertainty, groundtruth):
    prediction = [get_gp_prediction(pt[0]) for pt in uncertainty]
    weights = [get_certainty_share(pt[0], pt[1]) for pt in uncertainty]

    return (matthews_corrcoef(groundtruth, prediction, sample_weight=weights), matthews_corrcoef(groundtruth, prediction))


class WeightedMccExtractor(EvalMetricExtractor):

    def __init__(self):
        self.name = "Weighted Matthew Correlation Coefficient"

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        if "GpUncertainty" in get_all_keys(dictonary):
            uncertainty_measures = nested_lookup("GpUncertainty", dictonary)  # containts measures of both runs
            ground_truth = nested_lookup("GroundTruthLogger", dictonary)

            run_accuracy = []
            for uncert, gt in zip(uncertainty_measures, ground_truth):
                stepwise_accuracy = []
                for step_uncertainty in uncert:
                    stepwise_accuracy.append(weighted_accuracy(step_uncertainty, gt))
                run_accuracy.append(stepwise_accuracy)

            return run_accuracy

        elif "SvddUncertainty" in get_all_keys(dictonary):
            return []

        else:
            raise RuntimeError()
