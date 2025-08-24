import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from scipy.optimize import linear_sum_assignment

from .base_tb import TightBinding
from ...geometry import Geometry

from IPython import embed

class TightBindingEdge(TightBinding):
  
    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser) 
        self.location = "edge"  

    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building 'Edge' Hamiltonian...")
        self.sublattice_data_dict = sublattice_data_dict = self.sublattice_data(geometry)
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
        print(f"'Edge' Hamiltonian - Done.")

    def sublattice_data(self, geometry:Geometry):
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
            sublattice_data_dict[sub_label][idx] = self._sublattice_data(geometry, self.location, idx)
            sublattice_idxs.append(idx)
            path = sites[idx].copy()
            for i in range(geometry.N_r - 1):
                path += T_p
                try:
                    sublattice_n = np.where(np.all(np.isclose(sites, path, atol=1e-8), axis=1))[0][0]
                    sublattice_data_dict[sub_label][sublattice_n] = self._sublattice_data(geometry, self.location, sublattice_n)
                    sublattice_idxs.append(sublattice_n)
                except:
                    pass
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
            self.H_k_dict = H_k
        else:
            ValueError("Only 'real' and 'reciprocal' problems considered")
        print(f"'Edge' Eigenvalues - Done!")
        return perf_counter() - start

    def _fourier_transform(self, geometry:Geometry, k: int) -> np.ndarray:
        N_projections = len(self.coupled_states)
        N_sites = len(self.sublattice_idxs)
        N = N_sites * N_projections
        H_k = np.zeros(shape=(N, N), dtype=complex)
        # Build
        idx_map = {idx: pos for pos, idx in enumerate(self.sublattice_idxs)}
        for idx_i in self.sublattice_idxs:
            i = idx_map[idx_i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            site_dict_i = self.site_data_dict[idx_i]
            # NN Hoppings
            self._hoppings_ft(
                geometry, N_projections, idx_map, row_slice, idx_i, site_dict_i, H_k, k
            )
            # NNN Spin-Orbit Coupling
            self._kane_mele_coupling_ft(
                geometry, N_projections, idx_map, row_slice, idx_i, site_dict_i, H_k, k
            )
            # Chadi Spin-Orbit Coupling
            self._chadi_coupling_ft(
                row_slice, idx_i, site_dict_i, H_k
            )
            # Mean Field Interaction
            self._mean_field_interaction_ft(
                row_slice, idx_i, site_dict_i, H_k
            )
            # Staggered Sublattice Potential
            self._staggered_potential_ft(
                row_slice, idx_i, site_dict_i, H_k
            )
            # Zeeman Splitting
            self._zeeman_splitting_ft(
                row_slice, idx_i, site_dict_i, H_k
            )
        return H_k

    def _hoppings_ft(self, geometry:Geometry, N_projections, idx_map, row_slice, idx_i, site_dict_i, H_k:np.ndarray, k):
        phase_dict = geometry.get_phase_idxs(idx_i, site_dict_i["dm_dict_NN"], self.sublattice_idxs)
        for idx_j, idx_j_phase in phase_dict.items():
            j = idx_map[idx_j]
            col_slice = slice(j * N_projections, (j + 1) * N_projections)
            H_k_ij = 0
            # Site
            if idx_j in site_dict_i["NN_idxs"]:
                m_ij = site_dict_i["dm_dict_NN"][idx_j].copy()
                t_ij = site_dict_i["hopping_dict"][idx_j].copy()
                bloch_phase = np.exp(1j * k * m_ij)
                H_k_ij += bloch_phase * t_ij
            # Phase
            if idx_j_phase is not None:
                m_ij_phase = site_dict_i["dm_dict_NN"][idx_j_phase].copy()
                t_ij_phase = site_dict_i["hopping_dict"][idx_j_phase].copy()
                bloch_phase =  np.exp(1j * k * m_ij_phase)
                H_k_ij += bloch_phase * t_ij_phase
            H_k[row_slice, col_slice] += H_k_ij

    def _kane_mele_coupling_ft(self, geometry:Geometry, N_projections, idx_map, row_slice, idx_i, site_dict_i, H_k:np.ndarray, k):
        phase_dict = geometry.get_phase_idxs(idx_i, site_dict_i["dm_dict_NNN"], self.sublattice_idxs)
        for idx_j, idx_j_phase in phase_dict.items():
            j = idx_map[idx_j]
            col_slice = slice(j * N_projections, (j + 1) * N_projections)
            H_k_ij = 0
            # Site
            if idx_j in site_dict_i["NNN_idxs"]:
                m_ij = site_dict_i["dm_dict_NNN"][idx_j].copy()
                s_ij = site_dict_i["kane_mele_coupling_dict"][idx_j].copy()
                bloch_phase = np.exp(1j * k * m_ij)
                H_k_ij += bloch_phase * s_ij
            # Phase
            if idx_j_phase is not None:
                m_ij_phase = site_dict_i["dm_dict_NNN"][idx_j_phase].copy()
                s_ij_phase = site_dict_i["kane_mele_coupling_dict"][idx_j_phase].copy()
                bloch_phase =  np.exp(1j * k * m_ij_phase)
                H_k_ij += bloch_phase * s_ij_phase
            H_k[row_slice, col_slice] += H_k_ij
    
    def _chadi_coupling_ft(self, row_slice, idx_i, site_dict_i, H_k:np.ndarray):
        H_k_ii = 0
        c_ii = site_dict_i["chadi_coupling_dict"][idx_i]
        H_k_ii += c_ii
        H_k[row_slice, row_slice] += H_k_ii

    def _mean_field_interaction_ft(self, row_slice, idx_i, site_dict_i, H_k:np.ndarray):
        H_k_ii = 0
        u_ii = site_dict_i["mean_field_interaction_dict"][idx_i]
        H_k_ii += u_ii
        H_k[row_slice, row_slice] += H_k_ii

    def _staggered_potential_ft(self, row_slice, idx_i, site_dict_i, H_k:np.ndarray):
        H_k_ii = 0
        m_ii = site_dict_i["staggered_potential_dict"][idx_i]
        H_k_ii += m_ii
        H_k[row_slice, row_slice] += H_k_ii

    def _zeeman_splitting_ft(self, row_slice, idx_i, site_dict_i, H_k:np.ndarray):
        H_k_ii = 0
        z_ii = site_dict_i["zeeman_splitting_dict"][idx_i]
        H_k_ii += z_ii
        H_k[row_slice, row_slice] += H_k_ii

    def build_band_structure(self, geometry: Geometry):
        """
        Build continuous band structure for the edge by reordering eigenvalues
        based on eigenvector overlap between consecutive k-points.
        Returns:
            band_dict : reordered energies for each band
            path      : cumulative distance along k-path
        """
        k_edge = geometry.k_edge
        keys = [f"{k}" for k in k_edge]
        N_k = len(keys)
        N_bands = len(self.E_k_dict[keys[0]])
        E_ordered = np.zeros((N_k, N_bands))
        U_ordered = np.zeros((N_k, N_bands, N_bands), dtype=complex)
        permutation = [i for i in range(N_bands)]
        for i, key in enumerate(keys):
            E_k = self.E_k_dict[key]
            U_k = self.U_k_dict[key]
            if i == 0:
                E_ordered[0, :] = E_k
                U_ordered[0, :, :] = U_k 
                continue
            E_ordered[i, :] = E_k[permutation]
            U_k_ordered = U_k[:, permutation]
            U_ordered[i] = U_k_ordered
        band_dict = {n: E_ordered[:, n] for n in range(N_bands)}
        eigenvector_dict = {n: U_ordered[:, :, n] for n in range(N_bands)}
        self.band_structure_data = {
            "band_dict": band_dict,
            "eigenvector_dict": eigenvector_dict,
            "path": k_edge,
        }
    
    def get_edge_bands(self, geometry:Geometry, edge_sites=[0], k_target=0.0, threshold=0.28):
        """
        Return the list of band indices whose eigenvectors are edge localized
        """
        k_edge = geometry.k_edge
        N_sites = len(self.sublattice_idxs)
        N_projections = len(self.coupled_states)
        N_bands = N_sites * N_projections
        edge_bands = []
        k_idx = np.argmin(np.abs(k_edge - k_target))
        for i in edge_sites:
            for n in range(N_bands):
                weight = self.weight(k_idx, i, n)
                E_n = self.band_structure_data["band_dict"][n]
                if weight > threshold and np.any(np.abs(E_n) > 1e-5):
                    edge_bands.append(n)
        return sorted(np.unique(edge_bands))

    def plot_dispersion(self, geometry: Geometry, bands:list = [], edge_bands:list = [],
                        x_max:float=None, x_min:float=None, y_max:float=None, y_min:float=None,
                        mu:float=None, legend: bool = False, hide: bool = True) -> None:
        N_bands = len(self.sublattice_idxs) * len(self.coupled_states)
        if bands == []:
            bands = range(N_bands)
        E_list = self.band_structure_data["band_dict"]
        k_vals = self.band_structure_data["path"]
        plt.figure(figsize=(10, 8))
        for band in bands:
            E = E_list[band]
            if np.allclose(E, 0, rtol=1e-6) and hide:
                # Ignore zero values
                continue
            if band in edge_bands:
                color = "r"
            else:
                color = "b"
            plt.plot(k_vals, E, 
                    color=color, 
                    label=f"Band {band}")
        if mu is not None:
            plt.axhline(
                        mu,
                        color="k",
                        linestyle='--',
                        linewidth=1.5,
                    )
        plt.xlabel(r"$k_{\parallel}$")
        plt.ylabel("Energy (eV)")
        plt.title("Edge Band Structure")
        if legend:
            plt.legend()
        if x_min is not None or x_max is not None:
            plt.xlim(left=x_min, right=x_max)
        if y_min is not None or y_max is not None:
            plt.ylim(bottom=y_min, top=y_max)
        plt.show()

