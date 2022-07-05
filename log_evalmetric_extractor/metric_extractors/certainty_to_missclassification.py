from typing import List

from nested_lookup import nested_lookup, get_all_keys
import numpy as np

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


def get_uncertainty(mean, std):
    return np.divide(np.abs(mean), np.sqrt(2 * std))


def get_gp_result(mean, groundtruth) -> int:
    if (mean >= 0 and groundtruth == 1) or (mean < 0 and groundtruth == -1):
        return 0
    else:
        return 1


def get_svdd_scoring(dist, gt):
    if (dist <= 0 and gt == 1) or (dist > 0 and gt == -1):
        return 0
    else:
        return 1


def get_certainty_fraction_split(uncertainty, groundtruth, model):
    scores = []
    for i in range(len(uncertainty)):
        if model == "GP":
            certainty_fraction = get_uncertainty(uncertainty[i][0], uncertainty[i][1])
            score = get_gp_result(uncertainty[i][0], groundtruth[i])
        elif model == "SVDD":
            certainty_fraction = np.abs(uncertainty[i])
            score = get_svdd_scoring(uncertainty[i], groundtruth[i])
        else:
            raise RuntimeError
        scores.append([certainty_fraction, score])
    scores.sort(key=lambda x: x[0], reverse=True)
    splited = [[], [], [], [], []]
    size = len(scores) / 5
    for i in range(len(scores)):
        splited[np.minimum(int(i / size), 4)].append([scores[i][0], scores[i][1]])
    return [
        (np.average([i[0] for i in splited[0]]), np.average([i[1] for i in splited[0]])),
        (np.average([i[0] for i in splited[1]]), np.average([i[1] for i in splited[1]])),
        (np.average([i[0] for i in splited[2]]), np.average([i[1] for i in splited[2]])),
        (np.average([i[0] for i in splited[3]]), np.average([i[1] for i in splited[3]])),
        (np.average([i[0] for i in splited[4]]), np.average([i[1] for i in splited[4]])),
            ]


class UncertaintyMisclassificationCorrelation(EvalMetricExtractor):

    def __init__(self):
        self.name = "Uncertainty vs Misclassification"
        self.best_only = True

    def get_metrics_log(self, dictonary: dict) -> List[List]:
        ## gt and uncertainty
        if "GpUncertainty" in get_all_keys(dictonary):
            uncertainty_measures = nested_lookup("GpUncertainty", dictonary)  # containts measures of both runs
            ground_truth = nested_lookup("GroundTruthLogger", dictonary)

            run_uncertainty = []
            # iterate over both repeats
            for uncert, gt in zip(uncertainty_measures, ground_truth):
                run_uncertainty.append(get_certainty_fraction_split(uncert[len(uncert)-1], gt, "GP"))

            return run_uncertainty

        elif "SvddUncertainty" in get_all_keys(dictonary):
            uncertainty_measures = nested_lookup("SvddUncertainty", dictonary)  # containts measures of both runs
            ground_truth = nested_lookup("GroundTruthLogger", dictonary)

            run_uncertainty = []
            for uncert, gt in zip(uncertainty_measures, ground_truth):
                run_uncertainty.append(get_certainty_fraction_split(uncert[len(uncert)-1], gt, "SVDD"))

            return run_uncertainty

        else:
            raise RuntimeError()
