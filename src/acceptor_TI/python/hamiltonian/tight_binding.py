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
        self.available_terms = [
            "nearest_neighbour_hopping", "spin_orbit_interaction", "coulomb_interaction"
        ]

    def build_hamiltonian(self, cell_parser:CellParser, geometry:Geometry):
        print(f"Calculating eigenvalues...")
        #### FIXME: Only works for s-orbitals
        size = len(geometry.sites)
        k_vec = geometry.k_vec
        bulk_idx, neighbors_idx, dr_list = geometry.get_bulk_data()
        from IPython import embed; embed()
        t_s = -2.8  # eV
        Eplus  = np.zeros(len(k_vec), dtype=np.complex128)
        Eminus = np.zeros(len(k_vec), dtype=np.complex128)

        for i, k in enumerate(k_vec):
            # f(k) = sum_{NN} exp(ik · dr)
            f_k = sum(np.exp(1j * np.dot(k, dr)) for dr in dr_list)
            val = t_s * np.abs(f_k)
            Eplus[i]  =  val   # "upper" band
            Eminus[i] = -val   # "lower" band
        self.Eplus_map  = Eplus.reshape((size, size)).real
        self.Eminus_map = Eminus.reshape((size, size)).real
        print(f"Eigenvalues calculated.")

    def _nearest_neighbour_hopping(self):
        #TODO
        return

    def _coulomb_interaction(self):
        #TODO
        return

    def _change_term(self, term:str, eigenvalue:float=0, i:int=None, j:int=None)->None: 
        #TODO
        return

    def plot_dispersion(self, geometry: Geometry):
        print(f"Plotting dispersion...")
        u_grid = geometry.kx_grid
        v_grid = geometry.ky_grid
        Eplus_map  = self.Eplus_map
        Eminus_map = self.Eminus_map
        b1 = geometry.b1
        b2 = geometry.b2
        nx, ny = u_grid.shape
        kx_matrix = np.zeros_like(u_grid)
        ky_matrix = np.zeros_like(u_grid)

        # Convert each (u,v) -> (kx, ky)
        for i in range(nx):
            for j in range(ny):
                u = u_grid[i, j]
                v = v_grid[i, j]
                kxy = u * b1 + v * b2
                kx_matrix[i, j] = kxy[0]
                ky_matrix[i, j] = kxy[1]

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111, projection='3d')

        allE = np.concatenate([Eplus_map.ravel(), Eminus_map.ravel()])
        vmin, vmax = allE.min(), allE.max()

        surf1 = ax.plot_surface(kx_matrix, ky_matrix, Eplus_map,
                                cmap='coolwarm', vmin=vmin, vmax=vmax, alpha=0.8)
        surf2 = ax.plot_surface(kx_matrix, ky_matrix, Eminus_map,
                                cmap='coolwarm', vmin=vmin, vmax=vmax, alpha=0.8)

        m = plt.cm.ScalarMappable(cmap='coolwarm')
        m.set_array(allE)
        m.set_clim(vmin, vmax)
        fig.colorbar(m, ax=ax, pad=0.1, label="Energy (eV)")

        ax.set_title("Honeycomb NN Dispersion in the Actual BZ")
        ax.set_xlabel("k_x")
        ax.set_ylabel("k_y")
        ax.set_zlabel("E (eV)")
        plt.tight_layout()
        plt.show()
