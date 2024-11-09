import numpy as np
import json
import os
from types import SimpleNamespace
from typing import Union

# import kwant

from .parameter import Parameter

class CellParser:
    # TODO: This class will be dynamic in creating structures
    def __init__(self, data_path, file_name):
        # Material
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

    
