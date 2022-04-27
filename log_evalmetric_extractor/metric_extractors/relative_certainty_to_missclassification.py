from typing import List

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


def get_certainty_fraction(mean, std) -> int:
    share = get_certainty_share(mean, std)
    if share >= 0.9:
        return 0
    elif share >= 0.8:
        return 1
    elif share >= 0.7:
        return 2
    elif share >= 0.6:
        return 3
    elif share >= 0.5:
        return 4
    else:
        raise RuntimeError


def get_gp_result(mean, groundtruth) -> int:
    if (mean >= 0 and groundtruth == 1) or (mean < 0 and groundtruth == -1):
        return 0
    else:
        return 1


def get_avg(list):
    if len(list) == 0:
        return [0, 0]
    else:
        return [np.average(list), len(list)]


def certainty_split(uncertainty, groundtruth):
    split = ([], [], [], [], [])
    for i in range(len(uncertainty)):
        split[get_certainty_fraction(uncertainty[i][0], uncertainty[i][1])].append(get_gp_result(uncertainty[i][0], groundtruth[i]))

    return [get_avg(split[0]), get_avg(split[1]), get_avg(split[2]), get_avg(split[3]), get_avg(split[4])]  # TODO size


class RelativeCertaintyMisclassificationCorrelation(EvalMetricExtractor):

    def __init__(self):
        self.name = "Relative Certainty vs Misclassification"
        self.best_only = True

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        ## gt and uncertainty
        if "GpUncertainty" in get_all_keys(dictonary):
            uncertainty_measures = nested_lookup("GpUncertainty", dictonary)  # containts measures of both runs
            ground_truth = nested_lookup("GroundTruthLogger", dictonary)

            run_uncertainty = []
            # iterate over both repeats
            for uncert, gt in zip(uncertainty_measures, ground_truth):
                stepwise_uncert_dev = []
                # eval step_uncertainty
                for step_uncertainty in uncert:
                    ## Aufteilung je step in 0.8,0.6,0.5
                    stepwise_uncert_dev.append(certainty_split(step_uncertainty, gt))
                run_uncertainty.append(stepwise_uncert_dev)

            return run_uncertainty

        elif "SvddUncertainty" in get_all_keys(dictonary):
            return []

        else:
            raise RuntimeError()
