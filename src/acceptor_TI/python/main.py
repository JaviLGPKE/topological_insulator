import numpy as np
from matplotlib import pyplot as plt

# import kwant
# import quspin

from .cell_parser import CellParser
from .geometry import Geometry

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
        self.geometry = Geometry(cell_parser=self.cell_parser)
        