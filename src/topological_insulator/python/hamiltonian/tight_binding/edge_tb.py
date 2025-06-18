import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from collections import defaultdict
from scipy import linalg

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
        idxs_NN = [
            idx
            for sites_dict in sublattice_data_dict.values()
            for site_dict in sites_dict.values()
            for idx in site_dict["NN_idxs"]
        ]
        idxs_NNN = [
            idx
            for sites_dict in sublattice_data_dict.values()
            for site_dict in sites_dict.values()
            for idx in site_dict["NNN_idxs"]
        ]
        self.unique_idxs = unique_idxs = np.unique(
            np.concatenate([idxs_NN, idxs_NNN])
        )
        # Connectivity
        N_sites = len(unique_idxs)
        sublattice_connectivity = np.zeros(shape=(N_sites, N_sites))
        # Hamiltonian
        N_projections = self.n_projections
        H = np.zeros((N_sites * N_projections, N_sites * N_projections), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(unique_idxs)}
        for idx_i, site_dict_i in self.site_data_dict.items():
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            # Hoppings
            for idx_j in site_dict_i["NN_idxs"]:
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_ij = site_dict_i["hopping_dict"][idx_j].copy()
                sublattice_connectivity[i, j] = 1
                H[row_slice, col_slice] = H_ij 
                # h.c. 
                if idx_j not in self.sublattice_idxs:
                    sublattice_connectivity[j, i] = 1
                    H[col_slice, row_slice] = H_ij.conj().T
            # Spin-Orbit Coupling
            for idx_j in site_dict_i["NNN_idxs"]:
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_ij = site_dict_i["spin_orbit_coupling_dict"][idx_j].copy()
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
        # NOTE: Start from the bottom edge, so we need to go backwards
        # along the opposite direction of the descending basis vector
        T_p = a1 if a2[1] < a1[1] else a2
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

    def solve_eigenvalues(self, geometry:Geometry, H_type:str):
        print(f"Calculating 'Edge' eigenvalues...")
        start = perf_counter()
        if H_type == "real":
            H = self.H
            self.E, U = self._solve_eigenvalues(H)
            H_diag = U.conj().T @ H @ U
            tol = 1e-12 * geometry.lattice_constant
            self.H_diag = np.where(np.abs(H_diag) < tol, 0, H_diag)
        elif H_type in ["momentum", "reciprocal"]:
            E_k_dict, U_k_dict = {}, {}
            for k in geometry.k_edge:
                key = f"{k}"
                H_k = self._fourier_transform(geometry, k)
                E_k, U_k = self._solve_eigenvalues(H_k)
                E_k_dict[key] = E_k # Eigenvalues
                U_k_dict[key] = U_k # Eigenstates
            self.E_k_dict, self.U_k_dict = E_k_dict, U_k_dict
        else:
            ValueError("Only 'real' and 'reciprocal' problems considered")
        print(f"'Edge' Eigenvalues - Done!")
        return perf_counter() - start

    def _fourier_transform(self, geometry:Geometry, k: int) -> np.ndarray:
        N_projections = self.n_projections
        N_sites = len(self.sublattice_idxs)
        N = N_sites * N_projections
        H_k = np.zeros(shape=(N, N), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(self.sublattice_idxs)}
        for idx_i in self.sublattice_idxs:
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            site_dict_i = self.site_data_dict[idx_i]
            # Hoppings
            phase_dict = geometry._get_phase_idxs(idx_i, site_dict_i["dm_dict_NN"], self.sublattice_idxs)
            for idx_j, idx_j_phase in phase_dict.items():
                j = idx_map[idx_j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                # Physical
                neighbour_idxs = site_dict_i["NN_idxs"]
                if idx_j in neighbour_idxs:
                    m_ij = site_dict_i["dm_dict_NN"][idx_j]
                    t_ij = site_dict_i["hopping_dict"][idx_j].copy()
                else:
                    m_ij = site_dict_i["dm_dict_NNN"][idx_j]
                    dr = site_dict_i["dr_dict_NNN"][idx_j]
                    bond_length = np.linalg.norm(dr)
                    cosines = dr / bond_length
                    eigenvalue_dict = self._slater_koster_hoppings(geometry, idx_i, idx_j, cosines)
                    H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
                    t_ij = self.U.conj().T @ H_uncoupled @ self.U
                bloch_phase = np.exp(1j * k * m_ij)
                H_k_ij = bloch_phase * t_ij
                # Phase
                if idx_j_phase is not None:
                    m_ij_phase = site_dict_i["dm_dict_NN"][idx_j_phase]
                    t_ij_phase = site_dict_i["hopping_dict"][idx_j_phase].copy()
                    bloch_phase =  np.exp(1j * k * m_ij_phase)
                    H_k_ij += bloch_phase * t_ij_phase
                H_k[row_slice, col_slice] = H_k_ij
            # TODO: Spin-Orbit Coupling
        return H_k

    def plot_dispersion(self, geometry: Geometry, legend:bool=False, hide:bool=True) -> None:
        k_vals = np.array([float(key) for key in self.E_k_dict.keys()])
        k_vals_sorted = k_vals
        E_list = []
        for key in sorted(self.E_k_dict, key=lambda x: float(x)):
            E_k = self.E_k_dict[key]
            E_list.append(E_k)
        E_list = np.array(E_list)
        plt.figure(figsize=(10, 8))
        num_bands = E_list.shape[1]
        for band in range(num_bands):
            E = E_list[:, band]
            if np.allclose(E, 0, rtol=1e-6) and hide:
                # Ignore zero values
                continue 
            plt.plot(k_vals_sorted, E, label=f"Band {band}")
        plt.xlabel(r"$k_{\parallel}$")
        plt.ylabel("Energy")
        plt.title("Edge Band Structure")
        if legend:
            plt.legend()
        plt.show()
