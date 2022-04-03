import json
import os
from typing import List

from log_evalmetric_extractor.eval_metric_extractor import EvalMetricExtractor


def read_log(root_dir: str, metric_extractors: List[EvalMetricExtractor]) -> dict:
    logs = {}
    for mode in os.listdir(root_dir):
        curr_mode_dict = {}
        mode_dir = os.path.join(root_dir, mode)
        for dataset in os.listdir(mode_dir):
            curr_dataset_dict = {}
            dataset_dir = os.path.join(mode_dir, dataset)
            for model in os.listdir(dataset_dir):
                curr_model_dict = {}
                model_dir = os.path.join(dataset_dir, model)
                for qs in os.listdir(model_dir):
                    curr_qs_dict = {}
                    curr_qs_dir = os.path.join(model_dir, qs)
                    for sample in os.listdir(curr_qs_dir):

                        file_content = open(os.path.join(curr_qs_dir, sample))
                        try:
                            data_object = json.loads(file_content.read())
                        except Exception:
                            print("noo")

                        founds = {}
                        for metric_result in [metric.get_metric_as_dict(data_object) for metric in metric_extractors]:
                            founds[metric_result[0]] = metric_result[1]

                        best_k = sample.split("_")[0]
                        for founds_key in founds.keys():
                            if best_k in curr_qs_dict.keys():
                                if founds_key in curr_qs_dict[best_k].keys():
                                    curr_qs_dict[best_k][founds_key].append(founds[founds_key])
                                else:
                                    curr_qs_dict[best_k][founds_key] = [founds[founds_key]]
                            else:
                                curr_qs_dict[best_k] = {founds_key: [founds[founds_key]]}

                    curr_model_dict[qs] = curr_qs_dict
                curr_dataset_dict[model] = curr_model_dict
            curr_mode_dict[dataset] = curr_dataset_dict
        logs[mode] = curr_mode_dict
    return logs
