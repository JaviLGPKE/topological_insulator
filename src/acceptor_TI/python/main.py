import numpy as np
from matplotlib import pyplot as plt

from .model_options import ModelOptions
from .cell_parser import CellParser
from .geometry import Geometry
from .hamiltonian import (TightBinding, TightBindingBulk, TightBindingEdge, WaveFunction)

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.save_path = save_path
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
        self.hamiltonian = {
            "bulk": {
                "tight_binding": None,
                "wavefunction": None
            },
            "edge": {
                "tight_binding": None,
                "wavefunction": None
            }
        }
    
    def setup(self, N_r=10, N_k=200, location:str = "bulk", BZ:str="reduced"):
        assert(N_r >= 10)
        # Model Options
        self.model_options = ModelOptions(N_r, N_k, location, BZ)
        # Geometry
        self.geometry = Geometry(model_options=self.model_options, cell_parser=self.cell_parser)
        self.geometry.build_lattice()
        # Hamiltonian
        for key in self.hamiltonian.keys():
            # Tight-Binding Model
            if location not in [key, "both"]:
                continue
            TB = TightBindingBulk if key == "bulk" else TightBindingEdge
            self.hamiltonian[key]["tight_binding"] = TB(
                model_options=self.model_options, cell_parser=self.cell_parser)
            tight_binding:TightBinding = self.hamiltonian[key]["tight_binding"]
            tight_binding.build_hamiltonian(geometry=self.geometry)
            # WaveFunction
            # wavefunction = WaveFunction(cell_parser=self.cell_parser)
            # self.Hamiltonian[location]["wavefunction"] = wavefunction.build_wavefunction()
    
    def run(self, acceptor:bool = False, H_type="real_space"):
        if acceptor:
            # TODO: implement through cell_parser
            # self.geometry.update_geometry() 
            # self.tight_biding.update_data() 
            ValueError("Acceptor case not implemented!")
        for key in self.hamiltonian.keys():
            if self.model_options.location not in [key, "both"]:
                continue
            tight_binding:TightBinding = self.hamiltonian[key]["tight_binding"]
            tight_binding.solve_eigenvalues(self.geometry, acceptor, H_type)
    
    def plot(self, plot_type="lattice", location:str=None):
        if plot_type == "lattice":
            self.geometry.plot_lattice()
        elif plot_type == "dispersion":
            tight_binding:TightBinding = self.hamiltonian[location]["tight_binding"]
            tight_binding.plot_dispersion(self.geometry)
            