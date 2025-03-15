import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from collections import defaultdict

from .base_tb import TightBinding
from ...geometry import Geometry

from IPython import embed

class TightBindingEdge(TightBinding):  
    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "edge"  

    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building 'Edge' Hamiltonian...")
        self.sublattice_data_dict = sublattice_data_dict = self._sublattice_data(geometry)
        self.site_data_dict = {k: v for d in sublattice_data_dict.values() for k, v in d.items()}
        idxs = [
            idx
            for sites_dict in sublattice_data_dict.values()
            for site_dict in sites_dict.values()
            for idx in site_dict["neighbour_idxs"]
        ]
        self.unique_idxs = unique_idxs = np.unique(np.array(idxs)) # NOTE: not unique 
        # Connectivity
        N_sites = len(unique_idxs)
        sublattice_connectivity = np.zeros(shape=(N_sites, N_sites))
        # Hamiltonian
        N_projections = self.n_orbitals * self.n_spins
        H = np.zeros((N_sites * N_projections, N_sites * N_projections), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(unique_idxs)}
        for idx_i, site_dict_i in self.site_data_dict.items():
            if idx_i not in idx_map:
                    continue
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            for idx_j in site_dict_i["neighbour_idxs"]:
                if idx_j not in idx_map:
                    continue
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_ij = site_dict_i["hopping_dict"][idx_j]
                sublattice_connectivity[i, j] = 1
                H[row_slice, col_slice] = H_ij 
                # h.c. 
                if idx_j not in self.sublattice_idxs:
                    sublattice_connectivity[j, i] = 1
                    H[col_slice, row_slice] = H_ij.conj().T
        self.sublattice_connectivity = sublattice_connectivity
        self.H = H
        print(f"'Edge' Hamiltonian - Done.")

    def _sublattice_data(self, geometry:Geometry):
        self.edge_idxs = edge_idxs = geometry.get_sublattice_idxs(self.location)
        sites = geometry.sites
        a1, a2 = geometry.a1, geometry.a2 
        T_p = a1 if a2[1] < a1[1] else a2
        # T_p[0] *= 1 # y-axis reflection
        # NOTE: Start from the bottom edge, so we need to go backwards
        # along the opposite direction of the descending basis vector
        sublattice_idxs = []
        sublattice_data_dict = {}
        for i, idx in enumerate(edge_idxs):
            sub_label = geometry.sublattice_labels[geometry.sublattice_label_idxs[idx]]
            sublattice_data_dict[sub_label] = {}
            sublattice_data_dict[sub_label][idx] = self.sublattice_data(geometry, self.location, idx)
            sublattice_idxs.append(idx)
            path = sites[idx].copy()
            for _ in range(geometry.N_r - 1):
                path += T_p
                sublattice_n = np.where(np.all(np.isclose(sites, path, atol=1e-8), axis=1))[0][0]
                sublattice_data_dict[sub_label][sublattice_n] = self.sublattice_data(geometry, self.location, sublattice_n)
                sublattice_idxs.append(sublattice_n)
        self.sublattice_idxs = np.array(sorted(sublattice_idxs))
        assert(list(sublattice_data_dict.keys()) == geometry.sublattice_labels[:geometry.n_sublattices])
        return sublattice_data_dict

    def solve_eigenvalues(self, geometry:Geometry, acceptor:bool, H_type:str):
        print(f"Calculating 'Edge' eigenvalues...")
        start = perf_counter()
        if H_type == "real_space":
            H = self.H
            self.E, U = self._solve_eigenvalues(H)
            H_diag = U.conj().T @ H @ U
            tol = 1e-12 * geometry.lattice_constant
            self.H_diag = np.where(np.abs(H_diag) < tol, 0, H_diag)
        elif H_type == "reciprocal_space":
            E_k_dict = {}
            for k in geometry.k_edge:
                H_k = self._fourier_transform(geometry, k)
                E_k, _ = self._solve_eigenvalues(H_k)
                E_k_dict[f"{k}"] = E_k
            self.E_k_dict = E_k_dict
        else:
            ValueError("Only 'real' and 'reciprocal' problems considered")
        print(f"'Edge' Eigenvalues - Done.")
        return perf_counter() - start

    def _fourier_transform(self, geometry:Geometry, k: np.ndarray) -> np.ndarray:
        N_projections = self.n_orbitals * self.n_spins
        N_sites = len(self.sublattice_idxs)
        dims = N_sites * N_projections
        C_k = np.zeros(shape=(N_sites, N_sites), dtype=complex)
        H_k = np.zeros(shape=(dims, dims), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(self.sublattice_idxs)}
        for idx_i in self.sublattice_idxs:
            site_dict_i = self.site_data_dict[idx_i]
            if idx_i not in idx_map:
                continue
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            # Iterate of bonds with corresponding phases
            phase_dict = geometry._get_phase_idxs(idx_i, site_dict_i["dm_dict"], self.sublattice_idxs)
            # embed()
            for idx_j, phase_idx_j in phase_dict.items():
                H_ij_k = site_dict_i["hopping_dict"][idx_j].copy()
                C_ij_k = 1
                if phase_idx_j is not None:
                    H_ij = site_dict_i["hopping_dict"][phase_idx_j]
                    m_ij = site_dict_i["dm_dict"][phase_idx_j]
                    # T is equivalent to one basis vector, hence the translation will
                    # correspond to 2 sublattices with equal labels
                    bloch_phase =  np.exp(2j * k * m_ij) 
                    H_ij_k += H_ij * bloch_phase
                    C_ij_k += bloch_phase
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_k[row_slice, col_slice] = H_ij_k
                C_k[i, j] = C_ij_k
        if self.model_options.solve_connectivity:
            return C_k
        else:
            return H_k

    def plot_dispersion(self, geometry: Geometry) -> None:
        k_vals = np.array([float(key) for key in self.E_k_dict.keys()])
        k_vals_sorted = k_vals
        E_list = []
        for key in sorted(self.E_k_dict, key=lambda x: float(x)):
            E_k = self.E_k_dict[key]
            E_list.append(E_k)
        E_list = np.array(E_list)
        plt.figure(figsize=(8, 6))
        num_bands = E_list.shape[1]
        for band in range(num_bands):
            E = E_list[:, band]
            if np.allclose(E, 0, rtol=1e-12):
                # Ignore zero values
                continue 
            plt.plot(k_vals_sorted, E, label=f"Band {band}")
        plt.xlabel(r"$k_{\parallel}$")
        plt.ylabel("Energy")
        plt.title("Edge Band Structure")
        # plt.legend()
        plt.show()
