from typing import List

from nested_lookup import nested_lookup, get_all_keys
import numpy as np

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


def get_certainty_share(mean, std):
    return np.divide(np.abs(mean), np.sqrt(2 * std))


def get_certainty_fraction(mean, std, splits) -> int:
    share = get_certainty_share(mean, std)
    if share >= splits[0]:
        return 0
    elif share >= splits[1]:
        return 1
    elif share >= splits[2]:
        return 2
    elif share >= splits[3]:
        return 3
    elif share >= splits[4]:
        return 4
    else:
        raise RuntimeError


def get_gp_result(mean, groundtruth) -> int:
    if (mean >= 0 and groundtruth == 1) or (mean < 0 and groundtruth == -1):
        return 1
    else:
        return 0


def get_splits(uncertainty):
    shares = [get_certainty_share(single[0], single[1]) for single in uncertainty]
    shares.sort(reverse=True)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    splits = [int(len(shares) * i) for i in fractions]
    return [shares[np.maximum(0, i - 1)] for i in splits]


def certainty_split(uncertainty, groundtruth):
    # 0.1, 0,2, 0.3, 0.4, else..
    split_intervals = get_splits(uncertainty)
    split = ([], [], [], [], [])
    for i in range(len(uncertainty)):
        split[get_certainty_fraction(uncertainty[i][0], uncertainty[i][1], split_intervals)].append(get_gp_result(uncertainty[i][0], groundtruth[i]))

    return [np.average(split[0], axis=0),
            np.average(split[1], axis=0),
            np.average(split[2], axis=0),
            np.average(split[3], axis=0),
            np.average(split[4], axis=0)]


######## svdd

def get_svdd_split(uncertaity):
    shares = [np.abs(x) for x in uncertaity]
    shares.sort(reverse=True)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    splits = [int(len(shares) * i) for i in fractions]
    return [shares[np.maximum(0, i - 1)] for i in splits]


def get_svdd_certainty_fraction(dist, splits) -> int:
    if dist >= splits[0]:
        return 0
    elif dist >= splits[1]:
        return 1
    elif dist >= splits[2]:
        return 2
    elif dist >= splits[3]:
        return 3
    elif dist >= splits[4]:
        return 4
    else:
        raise RuntimeError


def get_svdd_scoring(dist, gt):
    if (dist <= 0 and gt == 1) or (dist > 0 and gt == -1):
        return 1
    else:
        return 0


def certainty_split_svdd(uncertainty, groundtruth):  # TODO duble check if inlier have negative distance
    split = ([], [], [], [], [])
    splits = get_svdd_split(uncertainty)

    for i in range(len(uncertainty)):
        split[get_svdd_certainty_fraction(np.abs(uncertainty[i]), splits)].append(get_svdd_scoring(np.abs(uncertainty[i]), groundtruth[i]))

    return [np.average(split[0], axis=0),
            np.average(split[1], axis=0),
            np.average(split[2], axis=0),
            np.average(split[3], axis=0),
            np.average(split[4], axis=0)]


class UncertaintyMisclassificationCorrelation(EvalMetricExtractor):

    def __init__(self):
        self.name = "Uncertainty vs Misclassification"

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
            uncertainty_measures = nested_lookup("SvddUncertainty", dictonary)  # containts measures of both runs
            ground_truth = nested_lookup("GroundTruthLogger", dictonary)

            run_uncertainty = []
            for uncert, gt in zip(uncertainty_measures, ground_truth):
                stepwise_uncert_dev = []
                # eval step_uncertainty
                for step_uncertainty in uncert:
                    ## Aufteilung je step in tp fp fn tn
                    stepwise_uncert_dev.append(certainty_split_svdd(step_uncertainty, gt))
                run_uncertainty.append(stepwise_uncert_dev)

            return run_uncertainty

        else:
            raise RuntimeError()
