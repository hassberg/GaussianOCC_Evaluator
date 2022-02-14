from nested_lookup import nested_lookup
import json


class LogfileReader:

    def __init__(self, logfiles_by_experiments: [str, str, [str]]):
        self.available_logs = logfiles_by_experiments

    def get_logs_by_experiments(self):
        evaluation_objects = []

        for x in self.available_logs:
            run_evaluation = []
            for file in x[2]:
                file_content = open(file)
                data_object = json.loads(file_content.read())
                for name in x[1]:
                    lookup_result = nested_lookup(name, data_object, wild=True, with_keys=True)
                    for obj in lookup_result.values():
                        run_evaluation.append(obj[0])
            evaluation_objects.append((x[0], run_evaluation))

        return evaluation_objects
