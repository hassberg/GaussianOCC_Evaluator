from logfile_reader.logfile_reader import LogfileReader
from logfile_evaluation_metrics.initial_correctness import InitialCorrectnessMeasure
from logfile_evaluation_metrics.ramp_up import RampUpMeasure
from logfile_evaluation_metrics.quality_range import QualityRangeMeasure
from logfile_evaluation_metrics.average_end_quality import AverageEndQualityMeasure
from logfile_evaluation_metrics.learning_stability import LearningStabilityMeasure
from logfile_evaluation_metrics.certainty_reached import CertaintyMeasure
from logfile_evaluation_metrics.outlier_sample_rate import OutlierSamplingMeasure

from log_evaluator import LogEvaluator

file_name = ".\\logfiles\\data_log_mean_compare_tries_30"

experiments_run = [("Constant, Uncertainty", ["ConstantPriorUncertaintyBasedBP"], [file_name]),
                   ("Constant, Mean", ["ConstantPriorMeanBasedBP"], [file_name]),
                   ("Custom, Uncertainty", ["CustomPriorUncertaintyBasedBP"], [file_name]),
                   ("Custom, Mean", ["CustomPriorMeanBasedBP"], [file_name])]
logfile_reader = LogfileReader(experiments_run)
logs = logfile_reader.get_logs_by_experiments()

evaluator = LogEvaluator([InitialCorrectnessMeasure(), QualityRangeMeasure(), RampUpMeasure(), AverageEndQualityMeasure(), LearningStabilityMeasure(), OutlierSamplingMeasure(), CertaintyMeasure()], logs)
evaluator.evaluate("mean_compare_numTries_30")
