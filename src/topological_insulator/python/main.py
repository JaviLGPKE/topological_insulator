from .model_options import ModelOptions
from .cell_parser import CellParser
from .geometry import Geometry
from .hamiltonian import (TightBinding, TightBindingBulk, TightBindingEdge, TopologicalInvariants)

class Problem:
    def __init__(self, data_path:str, file_name:str, save_path=None):
        self.save_path = save_path
        self.cell_parser = CellParser(data_path=data_path, file_name=file_name)
        self.hamiltonian = {
            "bulk": {
                "tight_binding": None,
                "topological_invariants": None
            },
            "edge": {
                "tight_binding": None,
                "topological_invariants": None
            }
        }
    
    def setup(self, N_r=10, N_k=200, location:str = "bulk", BZ:str="reduced", dangling_bonds:bool=False):
        if location not in ["both", "edge", "bulk"]:
            raise ValueError("Only 'bulk' and 'edge' cases considered.")
        assert(N_r >= 10)
        # Model Options
        self.model_options = ModelOptions(N_r, N_k, location, BZ, dangling_bonds)
        # Geometry
        self.geometry = Geometry(model_options=self.model_options, cell_parser=self.cell_parser)
        self.geometry.build_lattice()
        # Hamiltonian
        for key in self.hamiltonian.keys():
            # Tight-Binding Approximation
            if location not in [key, "both"]:
                continue
            TB = TightBindingBulk if key == "bulk" else TightBindingEdge
            self.hamiltonian[key]["tight_binding"] = TB(
                model_options=self.model_options, cell_parser=self.cell_parser)
            tight_binding:TightBinding = self.hamiltonian[key]["tight_binding"]
            tight_binding.build_hamiltonian(geometry=self.geometry)
    
    def run(self, H_type="real"):
        location = self.model_options.location
        # Hamiltonian
        for key in self.hamiltonian.keys():
            if location not in [key, "both"]:
                continue
            # Tight-Binding
            tight_binding:TightBinding = self.hamiltonian[key]["tight_binding"]
            tight_binding.solve_eigenvalues(self.geometry, H_type)
            # invariants
            invariants = TopologicalInvariants(
                model_options=self.model_options, cell_parser=self.cell_parser, 
                geometry=self.geometry, tight_binding = tight_binding
            )
            self.hamiltonian[key]["topological_invariants"] = invariants
    
    def get_topological_invariant(self, band = 0, tol= 1e-6):
        location = self.model_options.location
        assert(location in ["both", "bulk"])
        invariants: TopologicalInvariants = self.hamiltonian["bulk"]["topological_invariants"]
        return invariants.get_topological_invariant(band, tol)

    def plot(self, plot_type="lattice", location:str=None, legend:bool=False, hide:bool=True, F=None):
        if plot_type == "lattice":
            self.geometry.plot_lattice()
        elif plot_type == "dispersion":
            tight_binding:TightBinding = self.hamiltonian[location]["tight_binding"]
            tight_binding.plot_dispersion(self.geometry, legend=legend, hide=hide)
        elif plot_type == "high_symmetry":
            assert(location == "bulk")
            tight_binding:TightBinding = self.hamiltonian[location]["tight_binding"]
            tight_binding.plot_band_structure(self.geometry, hide=hide)
        elif plot_type in ["berry_flux", "berry_curvature"]:
            invariants:TopologicalInvariants = self.hamiltonian[location]["topological_invariants"]
            invariants.plot_berry_flux(F)
            