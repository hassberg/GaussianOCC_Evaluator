from typing import List

from nested_lookup import nested_lookup, get_all_keys
import numpy as np

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


def get_certainty_fraction(mean, std) -> int:
    upper = mean + 2 * std
    lower = mean - 2 * std

    if (upper >= 0 and lower >= 0) or (upper < 0 and lower < 0):
        return 0
    else:
        uncertainty = np.absolute(lower) + np.absolute(upper)
        share = np.maximum(np.abs(upper), np.abs(lower)) / uncertainty
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
        return 1
    else:
        return 0


def certainty_split(uncertainty, groundtruth):
    # 0.1, 0,2, 0.3, 0.4, else..
    split = ([], [], [], [], [])
    for i in range(len(uncertainty)):
        split[get_certainty_fraction(uncertainty[i][0], uncertainty[i][1])].append(get_gp_result(uncertainty[i][0], groundtruth[i]))

    return [(np.average(split[0], axis=0), len(split[0])),
            (np.average(split[1], axis=0), len(split[1])),
            (np.average(split[2], axis=0), len(split[2])),
            (np.average(split[3], axis=0), len(split[3])),
            (np.average(split[4], axis=0), len(split[4]))]


def confusion_split_svdd(uncertainty, groundtruth):  # TODO duble check if inlier have negative distance
    split = ([], [], [], [])
    for i in range(len(uncertainty)):
        if uncertainty[i] <= 0 and groundtruth[i] == 1:
            # tp
            split[0].append(uncertainty[i])
        elif uncertainty[i] <= 0 and groundtruth[i] == -1:
            # fp
            split[1].append(uncertainty[i])
        elif uncertainty[i] > 0 and groundtruth[i] == 1:
            # fn
            split[2].append(uncertainty[i])
        elif uncertainty[i] > 0 and groundtruth[i] == -1:
            # tn
            split[3].append(uncertainty[i])
        else:
            print("hää")
    # average mean, uncertainty je step
    return [np.average(split[0], axis=0), np.average(split[1], axis=0), np.average(split[2], axis=0), np.average(split[3], axis=0)]


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
                    stepwise_uncert_dev.append(confusion_split_svdd(step_uncertainty, gt))
                run_uncertainty.append(stepwise_uncert_dev)

            return run_uncertainty

        else:
            raise RuntimeError()
