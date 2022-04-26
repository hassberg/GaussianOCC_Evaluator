def get_dataset_name(name: str):
    prefix = "Dataset: "
    if "glass-1,2,3,4" in name:
        return prefix + "Glass"
    elif ".." in name:
        return ".."
    else:
        raise RuntimeError


def get_model_name(name: str, req_prefix: bool = False):
    if req_prefix:
        prefix = "Model: "
    else:
        prefix = ""
    if "SVDDNegSurrogateModel" in name:
        return prefix + "SVDDneg"
    elif "CustomModelBasedPriorMeanSurrogateModel" in name:
        return prefix + "SVDDBasedMeanPriorGP"
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
