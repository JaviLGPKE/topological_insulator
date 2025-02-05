import numpy as np
from matplotlib import pyplot as plt

from .model_options import ModelOptions
from .cell_parser import CellParser
from .geometry import Geometry
from .hamiltonian import TightBinding

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
        self.geometry = Geometry(cell_parser=self.cell_parser)
    
    def setup(self):
        model_options = ModelOptions()
        self.geometry.build_lattice()
        self.tight_binding = TightBinding(model_options=model_options)
    
    def run(self):
        self.tight_binding.build_hamiltonian(
            cell_parser = self.cell_parser, 
            geometry = self.geometry
        )
    
    def plot(self, type="lattice"):
        if type == "lattice":
            self.geometry.plot_lattice()
        elif type == "dispersion":
            self.tight_binding.plot_dispersion(self.geometry)
            