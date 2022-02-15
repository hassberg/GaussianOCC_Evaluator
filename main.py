from logfile_reader.logfile_reader import LogfileReader
from logfile_evaluation_metrics.initial_correctness import InitialCorrectnessMeasure
from logfile_evaluation_metrics.ramp_up import RampUpMeasure
from logfile_evaluation_metrics.quality_range import QualityRangeMeasure
from logfile_evaluation_metrics.average_end_quality import AverageEndQualityMeasure
from logfile_evaluation_metrics.learning_stability import LearningStabilityMeasure

from log_evaluator import LogEvaluator

experiments_run = [("Variance selections", ["VarianceBasedSelectionBP"], [".\\logfiles\\data_log_multi"]),
                   ("Uncertainty sampling", ["UncertaintyBasedSelectionBP"], [".\\logfiles\\data_log_multi"]),
                   ("Mean Based", ["MeanBasedSelectionBP"], [".\\logfiles\\data_log_multi"])]
logfile_reader = LogfileReader(experiments_run)
logs = logfile_reader.get_logs_by_experiments()

evaluator = LogEvaluator([InitialCorrectnessMeasure(), QualityRangeMeasure(), RampUpMeasure(), AverageEndQualityMeasure(), LearningStabilityMeasure()], logs)
evaluator.evaluate()
