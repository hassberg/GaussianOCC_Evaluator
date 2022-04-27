def get_dataset_name(name: str):
    prefix = "Dataset: "
    if "breast-cancer-wisconsin-2" in name:
        return prefix + "Breast Cancer Wisconsin"
    elif "ecoli-cp" in name:
        return prefix + "Ecoli"
    elif "glass-1,2,3,4" in name:
        return prefix + "Glass"
    elif "ionosphere-g" in name:
        return prefix + "Ionosphere"
    elif "lymphography-2" in name:
        return prefix + "Lymphographie"
    elif "page-blocks-1" in name:
        return prefix + "Pageblocks"
    elif "svmguide1-0" in name:
        return prefix + "SVM-Guide1_0"
    elif "svmguide1-1" in name:
        return prefix + "SVM-Guide1_1"
    elif "waveform-0" in name:
        return prefix + "Waveform_0"
    elif "waveform-1" in name:
        return prefix + "Waveform_1"
    elif "wdbc-B" in name:
        return prefix + "wdbc"
    elif "wpbc-N" in name:
        return prefix + "wpbc"
    elif "yeast-CYT" in name:
        return prefix + "Yeast_CYT"
    elif "yeast-NUC" in name:
        return prefix + "Yeast_NUC"
    else:
        raise RuntimeError


def get_model_name(name: str, req_prefix: bool = False):
    if req_prefix:
        prefix = "Model: "
    else:
        prefix = ""
    if "SVDDNegSurrogateModel" in name:
        return prefix + "SVDDneg"
    elif "SelfTrainingCustomModelBasedPriorMeanSurrogateModel" in name:
        return prefix + "OptimizingSVDDBasedMeanPriorGP"
    elif "CustomModelBasedPriorMeanSurrogateModel" in name:
        return prefix + "SVDDBasedMeanPriorGP"
    elif "ConstantPriorMeanSurrogateModel" in name:
        return prefix + "ConstantZeroPriorMeanGP"
    else:
        raise RuntimeError


def get_qs_name(name: str, req_prefix: bool = False):
    if req_prefix:
        prefix = "Selection Strategy: "
    else:
        prefix = ""

    if "GpDecisionBoundaryFocusedQuerySelection" in name:
        return prefix + "Mean Based"
    elif "UncertaintyBasedQuerySelection" in name:
        return prefix + "Uncertainty Based"
    elif "RandomOutlierSamplingSelectionCriteria" in name:
        return prefix + "Random Outlier"
    elif "SvddDecisionBoundaryFocusedQuerySelection" in name:
        return prefix + "Decision Boundary"
    else:
        raise RuntimeError
