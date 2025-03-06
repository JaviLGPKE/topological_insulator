import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter

from .base_tb import TightBinding
from ...geometry import Geometry

from IPython import embed

class TightBindingEdge(TightBinding):  
    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "edge"  

    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building 'Edge' Hamiltonian...")
        self.sublattice_data_dict = self._sublattice_data(geometry)
        sublattice_data_dict:dict = self.sublattice_data_dict
        idxs = [
            idx
            for sites_dict in sublattice_data_dict.values()
            for site_dict in sites_dict.values()
            for idx in site_dict["neighbour_idxs"]
        ]
        self.unique_idxs = unique_idxs = np.unique(np.array(idxs))
        # Connectivity
        N_subs = len(self.unique_idxs)
        sublattice_connectivity = np.zeros(shape=(N_subs, N_subs))
        # Hamiltonian
        N_projections = self.n_orbitals * self.n_spins
        N_sites = len(unique_idxs)
        H = np.zeros((N_sites * N_projections, N_sites * N_projections), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(unique_idxs)}
        for sublattice_dict in sublattice_data_dict.values():
            for idx_i, site_dict in sublattice_dict.items():
                if idx_i not in idx_map:
                    continue
                i = idx_map[idx_i]
                row_slice = slice(i * N_projections, (i + 1) * N_projections)
                for idx_j in site_dict["neighbour_idxs"]:
                    if idx_j not in idx_map:
                        continue
                    j = idx_map[idx_j]
                    sublattice_connectivity[i, j] = 1
                    sublattice_connectivity[j, i] = 1 # h.c
                    col_slice = slice(j * N_projections, (j + 1) * N_projections)
                    H_ij:np.ndarray = site_dict["hopping_dict"][idx_j]
                    H[row_slice, col_slice] = H_ij
                    H[col_slice, row_slice] = H_ij.conj().T # h.c
        self.sublattice_connectivity = sublattice_connectivity
        self.H = H
        print(f"'Edge' Hamiltonian - Done.")

    def _sublattice_data(self, geometry:Geometry):
        self.edge_idxs = edge_idxs = geometry.get_sublattice_idxs(self.location)
        geometry._build_brillouine_zone(self.edge_idxs)
        sites = geometry.sites
        a1, a2 = geometry.a1, geometry.a2 
        a = a2 if a1[1] > a2[1] else a1
        # NOTE: we start from the bottom edge, so we need to go backwards
        # along the opposite direction of the descending basis vector
        sublattice_idxs = []
        sublattice_data_dict = {}
        for i, idx in enumerate(edge_idxs):
            sub_label = geometry.sublattice_labels[geometry.sublattice_label_idxs[idx]]
            sublattice_data_dict[sub_label] = {}
            sublattice_data_dict[sub_label][idx] = self.sublattice_data(geometry, self.location, idx)
            sublattice_idxs.append(idx)
            path = sites[idx].copy()
            for n in range(geometry.N_r - 1):
                path -= a
                sublattice_n = np.where(np.all(np.isclose(sites, path, atol=1e-8), axis=1))[0][0]
                sublattice_data_dict[sub_label][sublattice_n] = self.sublattice_data(geometry, self.location, sublattice_n)
                sublattice_idxs.append(sublattice_n)
        self.sublattice_idxs = sublattice_idxs
        assert(list(sublattice_data_dict.keys()) == geometry.sublattice_labels[:geometry.n_sublattices])
        return sublattice_data_dict

    def solve_eigenvalues(self, geometry:Geometry, acceptor:bool, H_type:str):
        print(f"Calculating 'Edge' eigenvalues...")
        start = perf_counter()
        if H_type == "real_space":
            H = self.H
            self.E = self._solve_eigenvalues(H)
        elif H_type == "reciprocal_space":
            E_k_dict = {}
            for k in geometry.k_edge:
                H_k = self._fourier_transform(k, geometry.T_hat, geometry.T_norm)
                E_k = self._solve_eigenvalues(H_k)
                E_k_dict[f"{k}"] = E_k
            self.E_k_dict = E_k_dict
        else:
            ValueError("Only 'real' and 'reciprocal' problems considered")
        print(f"'Edge' Eigenvalues - Done.")
        return perf_counter() - start

    def _fourier_transform(self, k: np.ndarray, T_hat, T_norm) -> np.ndarray:
        N_projections = self.n_orbitals * self.n_spins
        H_k = self.H.copy()
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(self.unique_idxs)}
        for sublattice_dict in self.sublattice_data_dict.values():
            for idx_i, site_dict in sublattice_dict.items():
                if idx_i not in idx_map:
                    continue
                i = idx_map[idx_i]
                row_slice = slice(i * N_projections, (i + 1) * N_projections)
                for idx_j in site_dict["neighbour_idxs"]:
                    if (idx_j not in idx_map) or (idx_j in self.sublattice_idxs):
                        continue
                    j = idx_map[idx_j]
                    col_slice = slice(j * N_projections, (j + 1) * N_projections)
                    m_ij = site_dict["dm_dict"][idx_j]
                    phase = np.exp(1j * k * m_ij * T_norm)
                    H_k[row_slice, col_slice] *= phase
                    H_k[col_slice, row_slice] *= phase.conj().T # h.c
        return H_k

    def plot_dispersion(self, geometry: Geometry) -> None:
        k_vals = np.array([float(key) for key in self.E_k_dict.keys()])
        sort_idx = np.argsort(k_vals)
        k_vals_sorted = k_vals[sort_idx]
        E_list = []
        for key in sorted(self.E_k_dict, key=lambda x: float(x)):
            E_k = self.E_k_dict[key]
            E_list.append(E_k)
        E_list = np.array(E_list)  # shape: (N_k, num_bands)
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
        plt.title("Dispersion Relation")
        # plt.legend()
        plt.show()
