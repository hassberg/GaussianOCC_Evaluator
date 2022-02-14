from logfile_reader.logfile_reader import LogfileReader
from logfile_evaluation_metrics.initial_correctness import InitialCorrectnessMeasure

from log_evaluator import LogEvaluator

experiments_run = [("GP with else selections", ["VarianceBasedSelectionBP", "MeanBasedSelectionBP"],
                    [".\\logfiles\\data_log", ".\\logfiles\\data_log_multi"]),
                   ("GP with uncertainty sampling", ["UncertaintyBasedSelectionBP"],
                    [".\\logfiles\\data_log", ".\\logfiles\\data_log_multi"]),
                   ("GP with custom mean", ["CustomMeanGaussianProcessBlueprint"],
                    [".\\logfiles\\data_log", ".\\logfiles\\data_log_multi"])]
logfile_reader = LogfileReader(experiments_run)
logs = logfile_reader.get_logs_by_experiments()

evaluator = LogEvaluator([InitialCorrectnessMeasure()], logs)
evaluator.evaluate()
