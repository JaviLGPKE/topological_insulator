import numpy as np
from matplotlib import pyplot as plt

from .model_options import ModelOptions
from .cell_parser import CellParser
from .geometry import Geometry
from .hamiltonian import TightBinding

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
    
    def setup(self, size=10, N_k=200, dispersion=True, band_structure=False):
        assert(size >= 10)
        # Model Options
        model_options = ModelOptions(size, N_k, dispersion, band_structure)
        # Geometry
        self.geometry = Geometry(model_options=model_options, cell_parser=self.cell_parser)
        self.geometry.build_lattice(size, N_k)
        # Tight-Binding
        self.tight_binding = TightBinding(model_options=model_options)
        self.tight_binding.setup(
            cell_parser=self.cell_parser, 
            geometry=self.geometry
        )
    
    def run(self):
        self.tight_binding.get_eigenvalues(
            cell_parser=self.cell_parser, 
            geometry=self.geometry
        )
    
    def plot(self, type="lattice"):
        if type == "lattice":
            self.geometry.plot_lattice()
        elif type == "dispersion":
            self.tight_binding.plot_dispersion(self.geometry)
        elif type == "band_structure":
            self.tight_binding.plot_band_structure(self.geometry)
            