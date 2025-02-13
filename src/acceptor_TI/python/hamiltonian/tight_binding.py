import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from ..cell_parser import CellParser
from ..model_options import ModelOptions
from ..geometry import Geometry

from IPython import embed

class TightBinding:
    """
    Tight-Binding approximation Hamiltonian that can include, nearest neighbour hopping, 
    spin-orbit coupling interaction and Coulomb repulsive interaction terms.
    """
    def __init__(self, model_options:ModelOptions, cell_parser: CellParser):
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.available_terms = [
            "nearest_neighbour_hopping", "spin_orbit_interaction", "coulomb_interaction"
        ]
        self.n_orbitals = 4

    def _shared(self, geometry: Geometry):
        self.k_space = np.array([geometry.kx_grid, geometry.ky_grid])
        bulk_idx = geometry.get_bulk_idx()
        neighbours_idx = geometry.get_neighbours_data(bulk_idx)
        return bulk_idx, neighbours_idx
    
    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building Hamiltonian Matrix..")
        self.data_dict = data_dict = self._sublattice_data(geometry)
        idxs = [idx for i in data_dict.values() for idx in i["neighbour_idxs"]]
        self.unique_idxs = np.unique(np.array(idxs))
        idx_map = {idx: pos for pos, idx in enumerate(self.unique_idxs)}
        # Real Space Hamiltonian
        k = self.k_space
        N = len(self.unique_idxs)
        H = np.zeros(shape=(N, N))
        for sublattice_dict in data_dict.values():
            idx_i = sublattice_dict["idx"]
            if not (idx_i in idx_map):
                # Check that the central index is in our mapping 
                # (it may not be if it never appeared as a neighbour)
                continue
            i = idx_map[idx_i]
            for idx_j in sublattice_dict["neighbour_idxs"]:
                if idx_j in idx_map:
                    j = idx_map[idx_j]
                    H[i, j] = 1
                    H[j, i] = 1 # Hermicity
        self.real_hamiltonian = H
        print(f"Hamiltonian Matrix - Done.")
    
    def _sublattice_data(self, geometry:Geometry):
        n_sub = geometry.n_sublattices
        bulk_idx, neighbour_idxs = self._shared(geometry)
        sublattice_idxs = [bulk_idx]
        sublattice_idxs.extend(
            [idx for i, idx in enumerate(neighbour_idxs) if i < (n_sub -1)]
        )
        data_dict = {}
        for i, idx in enumerate(sublattice_idxs):
            sub_label = geometry.sublattice_labels[geometry.sublattice_label_idxs[idx]]
            neighbour_idxs = geometry.get_neighbours_data(idx)
            dr_list = geometry.get_dr(idx, neighbour_idxs, type="list")
            directional_cosines = geometry.bond_orientation(dr_list)
            data_dict[sub_label] = {
                "idx": idx,
                "neighbour_idxs": neighbour_idxs,
                "dr_dict": geometry.get_dr(idx, neighbour_idxs, type="dict"),
                "hopping_dict": self._slater_koster(neighbour_idxs, directional_cosines)
            }
        # Check we are considering unique sublattices
        assert(list(data_dict.keys()) == geometry.sublattice_labels[:n_sub])
        return data_dict
    
    def _slater_koster(self, neighbour_idxs, directional_cosines):
        nn_parser = self.cell_parser.eigenvalues.nn_hopping.value
        t_ss = nn_parser["t_ss_sigma"]
        t_sp = nn_parser["t_sp_sigma"]
        t_pp_sigma = nn_parser["t_pp_sigma"]
        t_pp_pi = nn_parser["t_pp_pi"]
        # H_ij orbital hoppings
        n_z = 0 # TODO: implement bukcling angle in json
        H_ij = {}
        for neighbour_idx, cosines in zip(neighbour_idxs, directional_cosines):
            l_x, m_y = cosines
            t_sx = l_x * t_sp
            t_sy = m_y * t_sp
            t_sz = n_z * t_sp
            t_xx = (l_x**2 * t_pp_sigma) + ((1 - l_x**2) * t_pp_pi)
            t_yy = (m_y**2 * t_pp_sigma) + ((1 - m_y**2) * t_pp_pi)
            t_zz = (n_z**2 * t_pp_sigma) + ((1 - n_z**2) * t_pp_pi)
            t_xy = l_x * m_y * (t_pp_sigma - t_pp_pi)
            t_xz = l_x * n_z * (t_pp_sigma - t_pp_pi)
            t_yz = m_y * n_z * (t_pp_sigma - t_pp_pi)
            #NOTE: s are symmetric, p orbitals are antisymmetric -> Under spatial inversion:
            # s(-r) = s(r) and p(-r) = -p(r), hence <s|H|x> = -<x|H|s>
            H_ij[neighbour_idx] = np.array(
                [
                    [t_ss, t_sx, t_sy, t_sz], 
                    [- t_sx, t_xx, t_xy, t_xz],
                    [- t_sy, t_xy, t_yy, t_yz],
                    [- t_sz, t_xz, t_yz, t_zz]
                ]
            )
        return H_ij

    def solve_eigenvalues(self, geometry:Geometry, acceptor:bool, type:str):
        # TODO: perform j, m_j angular momentum coupled basis 
        # transformation using Clebsch-Gordan coefficients
        if type == "bulk":
            self._solve_eigenvalues(geometry, acceptor)
            self._fourier_transform()
        elif type == "analytical_bulk" and not acceptor:
            self._analytical_bulk_eigenvalues(geometry)
        else:
            ValueError("Not Implemented!")

    def _solve_eigenvalues(self, geometry: Geometry, acceptor: bool):
        print(f"Calculating eigenvalues...")
        n_alpha = self.n_orbitals
        N_sites = len(self.unique_idxs)
        H = np.zeros((N_sites * n_alpha, N_sites * n_alpha), dtype=complex)
        # Populate Hamiltonian 
        idx_map = {global_idx: pos for pos, global_idx in enumerate(self.unique_idxs)}
        for label, sublattice_dict in self.data_dict.items():
            idx_i = sublattice_dict["idx"]
            i = idx_map[idx_i]
            row_slice = slice(i * n_alpha, (i + 1) * n_alpha)
            for idx_j in sublattice_dict["neighbour_idxs"]:
                j = idx_map[idx_j]
                col_slice = slice(j * n_alpha, (j + 1) * n_alpha)
                # Assign the hopping 4x4 matrix block to the Hamiltonian
                H_ij = sublattice_dict["hopping_dict"][idx_j]
                H_ij = sublattice_dict["hopping_dict"][idx_j]
                H[row_slice, col_slice] = H_ij
                H[col_slice, row_slice] = H_ij.conj().T # Hermicity
        self.E, U = np.linalg.eigh(H)
        H_diag = U.conj().T @ H @ U
        tol = 1e-12 * geometry.lattice_constant
        self.H_diag, self.H = np.where(np.abs(H_diag) < tol, 0, H_diag), H
    
    def _fourier_transform(self, geometry:Geometry):
        # TODO: perform fourier transform
        # E = self.E
        # n = len(E) / self.unique_idxs
        # E_k = []
        print(f"Calculating eigenvalues...")
        # k = self.k_space
        # f_k = 0
        # n_alpha = self.n_orbitals
        # idx_map = {global_idx: pos for pos, global_idx in enumerate(self.unique_idxs)}
        # for label, sublattice_dict in self.data_dict.items():
        #     idx_i = sublattice_dict["idx"]
        #     i = idx_map[idx_i]
        #     row_slice = slice(i * n_alpha, (i + 1) * n_alpha)
        #     for idx_j in sublattice_dict["neighbour_idxs"]:
        #         j = idx_map[idx_j]
        #         col_slice = slice(j * n_alpha, (j + 1) * n_alpha)
        #         # Fourier Transform
        #         r_ij = sublattice_dict["dr_dict"][idx_j]
        # self.E_plus_map = np.abs(f_k)
        # self.E_minus_map = -1 * np.abs(f_k)
        print(f"Eigenvalues calculated.")
        embed()


    def _analytical_bulk_eigenvalues(self, geometry:Geometry):
        if not self.model_options.dispersion:
            return
        print(f"Calculating eigenvalues...")
        k = self.k_space
        f_k = 0
        sites = geometry.sites
        bulk_idx, neighbours_idx = self._shared(geometry)
        bulk_dr_list = [sites[n] - sites[bulk_idx] for n in neighbours_idx]
        directional_cosines = geometry.bond_orientation(bulk_dr_list)
        hoppings_dict = self.slater_koster(neighbours_idx, directional_cosines)
        for idx, dr in zip(neighbours_idx, bulk_dr_list):
            slater_koster = hoppings_dict[idx]
            for t_alpha in slater_koster.values():
                # Nearest-Neighbour Hopping in Reciprocal Space
                f_k += t_alpha * np.exp(1j * (k[0]*dr[0] + k[1]*dr[1]))
        self.E_plus_map = np.abs(f_k)
        self.E_minus_map = -1 * np.abs(f_k)
        print(f"Eigenvalues calculated.")
    
    def _visualise_matrix(self, M):
        plt.figure(figsize=(12, 5))
        cmap = plt.get_cmap('coolwarm')
        cmap.set_bad('white')
        M_masked = np.ma.masked_where(M == 0, M)
        plt.subplot(1, 2, 1)
        plt.imshow(M_masked.real, cmap=cmap)
        plt.title('Real Part')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(M_masked.imag, cmap=cmap)
        plt.title('Imaginary Part')
        plt.colorbar()
        plt.show()

    def plot_analytical_dispersion(self, geometry: Geometry):
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