import json
import os

from .parameter import Parameter

class CellParser:
    def __init__(self, data_path, file_name):
        self.load_data(data_path, file_name)

    def load_data(self, data_path, file_name):
        path = os.path.join(data_path, file_name)
        if os.path.exists(path):
            with open(path, 'r') as file:
                json_data: dict = json.load(file)
        else:
            raise ValueError("Data path does not exist!")

        for hyperparameter, values in json_data.items():
            setattr(self, hyperparameter, Parameter(hyperparameter, values))

    
