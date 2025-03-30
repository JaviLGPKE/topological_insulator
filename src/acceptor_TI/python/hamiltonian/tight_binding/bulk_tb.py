import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter

from .base_tb import TightBinding
from ...geometry import Geometry

from IPython import embed

class TightBindingBulk(TightBinding):    

    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "bulk"

    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building 'Bulk' Hamiltonian...")
        self.sublattice_data_dict = self._sublattice_data(geometry)
        sublattice_data_dict:dict = self.sublattice_data_dict
        idxs = [idx for i in sublattice_data_dict.values() for idx in i["neighbour_idxs"]]
        self.unique_idxs = np.unique(np.array(idxs))
        # Connectivity
        N_subs = len(self.unique_idxs)
        sublattice_connectivity = np.zeros(shape=(N_subs, N_subs))
        # Hamiltonian
        N_projections = self.n_orbitals * self.n_spins
        N_sites = len(self.unique_idxs)
        H = np.zeros((N_sites * N_projections, N_sites * N_projections), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(self.unique_idxs)}
        for sublattice_dict in sublattice_data_dict.values():
            idx_i = sublattice_dict["idx"]
            if idx_i not in idx_map:
                continue
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            H_ii = sublattice_dict["spin_orbit_coupling_dict"][idx_i].copy()
            H[row_slice, row_slice] = H_ii
            for idx_j in sublattice_dict["neighbour_idxs"]:
                if idx_j not in idx_map:
                    continue
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_ij = sublattice_dict["hopping_dict"][idx_j]
                H[row_slice, col_slice] = H_ij
                sublattice_connectivity[i, j] = 1
                if idx_map not in self.sublattice_idxs: # h.c
                    H[col_slice, row_slice] = H_ij.conj().T
                    sublattice_connectivity[j, i] = 1 
        self.sublattice_connectivity = sublattice_connectivity
        self.H = H
        print(f"'Bulk' Hamiltonian - Done.")

    def _sublattice_data(self, geometry:Geometry):
        self.sublattice_idxs = sublattice_idxs = geometry.get_sublattice_idxs(self.location)
        sublattice_data_dict = {}
        for i, idx in enumerate(sublattice_idxs):
            sub_label = geometry.sublattice_labels[geometry.sublattice_label_idxs[idx]]
            sublattice_data_dict[sub_label] = self.sublattice_data(geometry, self.location, idx)
        assert(list(sublattice_data_dict.keys()) == geometry.sublattice_labels[:geometry.n_sublattices])
        return sublattice_data_dict

    def solve_eigenvalues(self, geometry:Geometry, acceptor:bool, H_type:str):
        tol = 1e-12 * geometry.lattice_constant
        print(f"Calculating 'Bulk' eigenvalues...")
        start = perf_counter()
        if H_type == "real_space":
            H = self.H
            self.E, U = self._solve_eigenvalues(H)
            H_diag = U.conj().T @ H @ U
            tol = 1e-12 * geometry.lattice_constant
            self.H_diag = np.where(np.abs(H_diag) < tol, 0, H_diag)
        elif H_type == "reciprocal_space":
            E_k_dict = {}
            for k_x in geometry.kx_bulk:
                for k_y in geometry.ky_bulk:
                    k = np.array([k_x, k_y])
                    H_k = self._fourier_transform(geometry, k)
                    E_k, _ = self._solve_eigenvalues(H_k)
                    E_k_dict[f"[{k_x},{k_y}]"] = E_k
            self.E_k_dict = E_k_dict
        else:
            ValueError("Only 'real' and 'reciprocal' problems considered")
        print(f"'Bulk' Eigenvalues - Done.")
        return perf_counter() - start

    def _fourier_transform(self, geometry:Geometry, k: np.ndarray) -> np.ndarray:
        N_projections = self.n_orbitals * self.n_spins
        N_sites = len(self.sublattice_idxs)
        # Hamiltonian
        dims = N_sites * N_projections
        C_k = np.zeros(shape=(N_sites, N_sites), dtype=complex)
        H_k = np.zeros(shape=(dims, dims), dtype=complex)
        for i in range(N_sites):
            sublattice_i_label = geometry.label_mapper[i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            data = self.sublattice_data_dict[sublattice_i_label]
            sublattice_dict = self.get_sublattice_dict(geometry, N_sites, data, k)
            for j in range(N_sites):
                sublattice_j_label = geometry.label_mapper[j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                # # Diagonal elements
                # if i == j:
                #     H_k[row_slice, col_slice] = sublattice_dict[sublattice_j_label]["H_k_ij"]
                #     C_k[i, j] = sublattice_dict[sublattice_j_label]["C_k_ij"]
                # # Off-diagonal elements
                # else:
                H_k[row_slice, col_slice] = sublattice_dict[sublattice_j_label]["H_k_ij"]
                C_k[i, j] = sublattice_dict[sublattice_j_label]["C_k_ij"]
        if self.model_options.solve_connectivity:
            return C_k
        else:
            return H_k
    
    def get_sublattice_dict(self, geometry, N_sites, data, k):
        sublattice_dict = {geometry.label_mapper[n]: {
                "C_k_ij":0, "H_k_ij": 0} for n in range(N_sites)}
        # Diagonal
        idx_i = data["idx"]
        idx_i_label = geometry.get_label(idx_i)
        H_ii = data["spin_orbit_coupling_dict"][idx_i].copy()
        sublattice_dict[idx_i_label]["H_k_ij"] += H_ii
        # Off-Diagonal
        for idx_j in data["neighbour_idxs"]:
            idx_j_label = geometry.get_label(idx_j)
            r_ij = data["dr_dict"][idx_j].copy() 
            H_ij = data["hopping_dict"][idx_j].copy()
            bloch_phase = np.exp(1j * np.dot(k, r_ij))
            sublattice_dict[idx_j_label]["C_k_ij"] += bloch_phase
            sublattice_dict[idx_j_label]["H_k_ij"] += bloch_phase * H_ij
        return sublattice_dict

    def plot_dispersion(self, geometry: Geometry):  
        kx, ky = geometry.kx_bulk, geometry.ky_bulk
        n_kx, n_ky = len(kx), len(ky)
        E_k_list = []
        for k_x in kx:
            for k_y in ky:
                key = f"[{k_x},{k_y}]"
                E_k_list.append(self.E_k_dict[key])
        E_stacked = np.stack(E_k_list)  # Shape: (n_kx * n_ky, n_bands)
        E_3d = E_stacked.reshape(n_kx, n_ky, -1)
        n_bands = E_3d.shape[2]  # nº eigenvalues per k-point
        KX, KY = np.meshgrid(kx, ky, indexing='ij') 
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        for band in range(n_bands):
            E = E_3d[:, :, band]
            if np.allclose(E, 0, rtol=1e-12):
                # Ignore zero values
                continue 
            ax.plot_surface(
                KX, KY, E,
                cmap='viridis',
                alpha=0.6, 
                edgecolor='none'
            )
        ax.set_xlabel(r'$k_x$', fontsize=12)
        ax.set_ylabel(r'$k_y$', fontsize=12)
        ax.set_zlabel(r'$E$', fontsize=12)
        plt.title('Bulk Band Structure', fontsize=14)
        plt.show()
