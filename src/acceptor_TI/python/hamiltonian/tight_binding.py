import numpy as np
from matplotlib import pyplot as plt

from ..cell_parser import CellParser
from ..model_options import ModelOptions
from ..geometry import Geometry

class TightBinding:
    """
    Tight-Binding approximation Hamiltonian that can include, nearest neighbour hopping, 
    spin-orbit coupling interaction and Coulomb repulsive interaction terms.
    """
    def __init__(self, model_options:ModelOptions):
        self.model_options = model_options
        self.available_terms = [
            "nearest_neighbour_hopping", "spin_orbit_interaction", "coulomb_interaction"
        ]

    def slater_koster(self, cell_parser: CellParser, directional_cosines):
        nn_parser = cell_parser.eigenvalues.nn_hopping.value
        n = 0 # TODO: implement bukcling angle in json
        slater_koster_coefficients = {}
        for bond_idx, cosines in enumerate(directional_cosines):
            l, m = cosines
            slater_koster_coefficients[bond_idx] = {
                "t_s_s": nn_parser["t_ss_sigma"],
                "t_s_x": l * nn_parser["t_sp_sigma"],
                "t_x_x": l**2 * nn_parser["t_pp_sigma"] + (1 - l**2) * nn_parser["t_pp_pi"],
                "t_x_y": l * m * (nn_parser["t_pp_sigma"] - nn_parser["t_pp_pi"]),
                "t_x_z": l * n * (nn_parser["t_pp_sigma"] - nn_parser["t_pp_pi"])
            }
        return slater_koster_coefficients

    def setup(self, cell_parser: CellParser, geometry: Geometry):
        self.k = np.array([geometry.kx_grid, geometry.ky_grid])
        bulk_idx, neighbours_idx, self.dr_list = geometry.get_bulk_data()
        directional_cosines = geometry.bond_orientation(self.dr_list)
        self.slater_koster_coefficients = self.slater_koster(cell_parser, directional_cosines)
    
    def eigenvalues(self):
        if not self.model_options.dispersion:
            return
        print(f"Calculating eigenvalues...")
        # Tight-Binding Model
        # TODO: Implement:
        # 1) On-Site Energy
        # 2) Mean-Field Interaction
        f_k = 0
        for i, dr in enumerate(self.dr_list):
            slater_koster = self.slater_koster_coefficients[i]
            for t_alpha in slater_koster.values():
                # Nearest-Neighbour Hopping
                f_k += self.nearest_neighbour_hopping(self.k, dr, t_alpha)
        self.E_plus_map = np.abs(f_k)
        self.E_minus_map = -1 * np.abs(f_k)
        print(f"Eigenvalues calculated.")

    def nearest_neighbour_hopping(self, k, dr, t_alpha):
        return t_alpha * np.exp(1j * (k[0]*dr[0] + k[1]*dr[1]))

    def coulomb_interaction(self):
        #TODO
        return

    def change_term(self, term:str, eigenvalue:float=0, i:int=None, j:int=None)->None: 
        #TODO
        return
    
    def plot_dispersion(self, geometry: Geometry):
        assert(self.model_options.dispersion)
        print(f"Plotting dispersion...")
        kx_matrix = np.zeros_like(geometry.kx_grid)
        ky_matrix = np.zeros_like(geometry.ky_grid)
        # Convert grid (u,v) → (kx, ky)
        for i in range(geometry.kx_grid.shape[0]):
            for j in range(geometry.ky_grid.shape[1]):
                u = geometry.kx_grid[i, j]
                v = geometry.ky_grid[i, j]
                kxy = u*geometry.b1 + v*geometry.b2
                kx_matrix[i, j] = kxy[0]
                ky_matrix[i, j] = kxy[1]
        # Plot 3D surface
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')
        surf1 = ax.plot_surface(kx_matrix, ky_matrix, self.E_plus_map, cmap='coolwarm', alpha=0.8)
        surf2 = ax.plot_surface(kx_matrix, ky_matrix, self.E_minus_map, cmap='coolwarm', alpha=0.8)
        # Overlay high-symmetry path
        # ax.plot(geometry.k_path[:,0], geometry.k_path[:,1], self.Eplus_band, 
        #         color='black', linewidth=2, label='Γ→K→M→Γ')
        # ax.plot(geometry.k_path[:,0], geometry.k_path[:,1], self.Eminus_band, 
        #         color='black', linewidth=2)
        ax.set_xlabel("k_x")
        ax.set_ylabel("k_y")
        ax.set_zlabel("E (eV)")
        # plt.legend()
        plt.show()
    
    def plot_band_structure(self, geometry: Geometry):
        assert(self.model_options.band_structure)
        print("Plotting band structure...")
        N_k = geometry.N_k
        k_vec = geometry.k_path
        Eplus = self.E_plus_band  
        Eminus = self.E_minus_band 

        # Create a parametric x-axis for the k-path (Γ → K → M → Γ)
        k_path_length = np.arange(len(k_vec))
        plt.figure(figsize=(10, 6))
        plt.plot(k_path_length, Eplus, label='E+', color='blue')
        plt.plot(k_path_length, Eminus, label='E-', color='red')

        # Positions must be scalars
        positions = [
            0, 
            N_k**2, 
            2 * N_k**2, 
            3 * N_k**2 - 1
        ]
        plt.xticks(
            positions,
            ['Γ', 'K', 'M', 'Γ'],
            fontsize=12
        )

        plt.xlabel('High-Symmetry Path')
        plt.ylabel('Energy (eV)')
        plt.legend()
        plt.grid(True)
        plt.show()