def get_dataset_name(name: str):
    if "glass-1,2,3,4" in name:
        return "Glass"
    elif ".." in name:
        return ".."
    else:
        raise RuntimeError


def get_model_name(name: str):
    if "SVDDNegSurrogateModel" in name:
        return "SVDDneg"
    elif "CustomModelBasedPriorMeanSurrogateModel" in name:
        return "SVDDBasedMeanPriorGP"
    else:
        raise RuntimeError


def get_qs_name(name: str):
    if "GpDecisionBoundaryFocusedQuerySelection" in name:
        return "Mean Based"
    elif "UncertaintyBasedQuerySelection" in name:
        return "Uncertainty Based"
    elif "RandomOutlierSamplingSelectionCriteria" in name:
        return "Random Outlier"
    elif "SvddDecisionBoundaryFocusedQuerySelection" in name:
        return "Decision Boundary"
    else:
        raise RuntimeError
