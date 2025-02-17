import numpy as np
from matplotlib import pyplot as plt

from .model_options import ModelOptions
from .cell_parser import CellParser
from .geometry import Geometry
from .hamiltonian import TightBinding

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
    
    def setup(self, size=10, N_k=200):
        assert(size >= 10)
        # Model Options
        model_options = ModelOptions(size, N_k)
        # Geometry
        self.geometry = Geometry(model_options=model_options, cell_parser=self.cell_parser)
        self.geometry.build_lattice(size, N_k)
        # Tight-Binding
        self.tight_binding = TightBinding(model_options=model_options, cell_parser=self.cell_parser)
        self.tight_binding.build_hamiltonian(geometry=self.geometry)
    
    def run(self, acceptor:bool = False, H_type="real_space"):
        if acceptor:
            # TODO: implement through cell_parser
            # self.geometry.update_geometry() 
            # self.tight_biding.update_data() 
            ValueError("Acceptor case not implemented!")
        self.t_total = self.tight_binding.solve_eigenvalues(self.geometry, acceptor, H_type)
    
    def plot(self, plot_type="lattice"):
        if plot_type == "lattice":
            self.geometry.plot_lattice()
        elif plot_type == "dispersion":
            self.tight_binding.plot_dispersion(self.geometry)
            