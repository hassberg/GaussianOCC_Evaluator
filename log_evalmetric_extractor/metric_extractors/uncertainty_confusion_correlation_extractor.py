from typing import List

from nested_lookup import nested_lookup, get_all_keys
import numpy as np

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor

def get_uncertainty(mean, std):
    return np.divide(np.abs(mean), np.sqrt(2 * std))

def confusion_split(uncertainty, groundtruth):
    split = [[], [], [], []]
    for i in range(len(uncertainty)):
        if uncertainty[i][0] >= 0 and groundtruth[i] == 1:
            # tp
            split[0].append(get_uncertainty(uncertainty[i][0], uncertainty[i][1]))
        elif uncertainty[i][0] >= 0 and groundtruth[i] == -1:
            # fp
            split[1].append(get_uncertainty(uncertainty[i][0], uncertainty[i][1]))
        elif uncertainty[i][0] < 0 and groundtruth[i] == 1:
            # fn
            split[2].append(get_uncertainty(uncertainty[i][0], uncertainty[i][1]))
        elif uncertainty[i][0] < 0 and groundtruth[i] == -1:
            # tn
            split[3].append(get_uncertainty(uncertainty[i][0], uncertainty[i][1]))
        else:
            raise RuntimeError
    # average mean, uncertainty je step
    return split


def confusion_split_svdd(uncertainty, groundtruth):#TODO duble check if inlier have negative distance
    split = [[], [], [], []]
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
    return split


class UncertaintyConfusionCorrelationExtractor(EvalMetricExtractor):

    def __init__(self):
        self.name = "Uncertainty Confusion Correlation"
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
                    ## Aufteilung je step in tp fp fn tn
                    stepwise_uncert_dev.append(confusion_split(step_uncertainty, gt))
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
