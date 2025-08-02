import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from scipy.optimize import linear_sum_assignment

from .base_tb import TightBinding
from ...geometry import Geometry

from IPython import embed

class TightBindingBulk(TightBinding):    

    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "bulk"

    def build_hamiltonian(self, geometry:Geometry):
        print(f"Building 'Bulk' Hamiltonian...")
        self.sublattice_data_dict = self.sublattice_data(geometry)
        print(f"'Bulk' Hamiltonian - Done.")

    def sublattice_data(self, geometry:Geometry):
        self.sublattice_idxs = sublattice_idxs = geometry.get_sublattice_idxs(self.location)
        sublattice_data_dict = {}
        for i, idx in enumerate(sublattice_idxs):
            sub_label = geometry.sublattice_labels[geometry.sublattice_label_idxs[idx]]
            sublattice_data_dict[sub_label] = self._sublattice_data(geometry, self.location, idx)
        assert(list(sublattice_data_dict.keys()) == geometry.sublattice_labels[:geometry.n_sublattices])
        return sublattice_data_dict

    def solve_eigenvalues(self, geometry:Geometry, H_type:str):
        tol = 1e-12 * geometry.lattice_constant
        print(f"Calculating 'Bulk' Eigenvalues...")
        start = perf_counter()
        if H_type == "real":
            H = self.H
            self.E, U = self._solve_eigenvalues(H)
            H_diag = U.conj().T @ H @ U
            tol = 1e-12 * geometry.lattice_constant
            self.H_diag = np.where(np.abs(H_diag) < tol, 0, H_diag)
        elif H_type in ["momentum", "reciprocal"]:
            H_k_dict, E_k_dict, U_k_dict = {}, {}, {}
            for k_x in geometry.kx_bulk:
                for k_y in geometry.ky_bulk:
                    key = f"[{k_x}, {k_y}]"
                    k = np.array([k_x, k_y])
                    H_k = self._fourier_transform(geometry, k)
                    E_k, U_k = self._solve_eigenvalues(H_k)
                    H_k_dict[key] = H_k # Hamiltonian
                    E_k_dict[key] = E_k # Eigenvalues
                    U_k_dict[key] = U_k # Eigenstates
            self.H_k_dict = H_k_dict
            self.E_k_dict, self.U_k_dict = E_k_dict, U_k_dict
        else:
            ValueError("Only 'real' and 'reciprocal'/'momentum' problems considered")
        print(f"'Bulk' Eigenvalues - Done!")
        return perf_counter() - start

    def _fourier_transform(self, geometry:Geometry, k: np.ndarray) -> np.ndarray:
        N_projections = len(self.coupled_states)
        N_sites = len(self.sublattice_idxs)
        N = N_sites * N_projections
        H_k = np.zeros(shape=(N, N), dtype=complex)
        for i in range(N_sites):
            sublattice_i_label = geometry.label_mapper[i]
            row_slice = slice(i * N_projections, (i + 1) * N_projections)
            data = self.sublattice_data_dict[sublattice_i_label]
            sublattice_dict = self.get_sublattice_dict(geometry, data, k, N_sites)
            for j in range(N_sites):
                sublattice_j_label = geometry.label_mapper[j]
                col_slice = slice(j * N_projections, (j + 1) * N_projections)
                H_k[row_slice, col_slice] = sublattice_dict[sublattice_j_label]
        return H_k

    def get_sublattice_dict(self, geometry, data, k, N_sites):
        sublattice_dict = {geometry.label_mapper[n]: 0 for n in range(N_sites)}
        idx_i = data["idx"]
        label_i = geometry.get_label(idx_i)
        # Electron Tunelling
        for idx_j in data["NN_idxs"]:
            label_j = geometry.get_label(idx_j)
            r_ij = data["dr_dict_NN"][idx_j].copy() 
            t_ij = data["hopping_dict"][idx_j].copy()
            bloch_phase = np.exp(1j * np.dot(k, r_ij))
            sublattice_dict[label_j] += bloch_phase * t_ij
        # Kane-Mele Spin-Orbit Coupling
        for idx_j in data["NNN_idxs"]:
            label_j = geometry.get_label(idx_j)
            r_ij = data["dr_dict_NNN"][idx_j].copy()
            s_ij =  data["kane_mele_coupling_dict"][idx_j].copy()
            bloch_phase = np.exp(1j * np.dot(k, r_ij))
            sublattice_dict[label_j] += bloch_phase * s_ij
        # Chadi Spin-Orbit Coupling
        c_ij = data["chadi_coupling_dict"][idx_i].copy()
        sublattice_dict[label_i] += c_ij
        # Mean Field Interaction
        u_ij = data["mean_field_interaction_dict"][idx_i].copy()
        sublattice_dict[label_i] += u_ij
        # Staggered Sublattice Potential
        m_ij = data["staggered_potential_dict"][idx_i].copy()
        sublattice_dict[label_i] += m_ij
        # Zeeman-Splitting
        z_ij = data["zeeman_splitting_dict"][idx_i].copy()
        sublattice_dict[label_i] += z_ij
        return sublattice_dict 

    def build_band_structure(self, geometry: Geometry):
        """
        Extract band energies along the high-symmetry path, ensuring continuous bands
        by reordering eigenvalues/eigenvectors based on eigenvector overlap.
        Returns:
            band_dict : dictionary containing energies for each band
            path      : high symmetry k-path
            ticks     : positions for labels
            labels    : high-symmetry labels
        """
        kpoints, path, ticks, labels = self._build_high_symmetry_path(geometry)
        kx_grid, ky_grid = geometry.kx_bulk, geometry.ky_bulk
        E_3d = self._reshape_Ek_into_grid(kx_grid, ky_grid)
        # Find grid indices for each k-point in the path
        indices = []
        for (kx, ky) in kpoints:
            ix = np.argmin(np.abs(kx_grid - kx))
            iy = np.argmin(np.abs(ky_grid - ky))
            indices.append((ix, iy))
        n_k = len(indices)
        N_bands = E_3d.shape[2]
        # Band Structure Correction 
        E_ordered = np.zeros((n_k, N_bands))
        U_ordered = np.zeros((n_k, N_bands, N_bands), dtype=complex)
        U_prev = None
        for i, (ix, iy) in enumerate(indices):
            key = f"[{kx_grid[ix]}, {ky_grid[iy]}]"
            E_k = E_3d[ix, iy, :]
            U_k = self.U_k_dict[key]
            if i == 0:
                E_ordered[0, :] = E_k
                U_prev = U_k
                continue
            # <Psi_{prev,i} | Psi_{current,j}>
            G = U_prev.conj().T @ U_k
            cost_matrix = 1 - np.abs(G) # Cost matrix for assignment
            # Find optimal assignment to maximize total overlap
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            permutation = col_ind
            E_ordered[i, :] = E_k[permutation]
            U_k_ordered = U_k[:, permutation]
            U_ordered[i] = U_k_ordered
            U_prev = U_k_ordered
        # Re-order in terms of mean energy
        band_dict = {i: E_ordered[:, i] for i in range(N_bands)}
        eigenvector_dict = {i: U_ordered[:, i, :] for i in range(N_bands)}
        mean_energies = {band_index: energies.mean()
                     for band_index, energies in band_dict.items()}
        sorted_bands = sorted(mean_energies, key=lambda i: mean_energies[i])
        band_dict = {new_idx: band_dict[old_idx]
                            for new_idx, old_idx in enumerate(sorted_bands)}
        eigenvector_dict = {new_idx: eigenvector_dict[old_idx]
                                    for new_idx, old_idx in enumerate(sorted_bands)}
        self.band_structure_data = {
            "band_dict": band_dict,
            "eigenvector_dict": eigenvector_dict,
            "path": path,
            "ticks": ticks,
            "labels": labels
        }
    
    def _build_high_symmetry_path(self, geometry):
        """
        Construct k-points along the high-symmetry path and corresponding distances,
        ticks and labels for plotting.
        Returns:
            kpoints : list of (kx, ky) tuples
            dist    : list of cumulative distances along path
            ticks   : positions for high-symmetry labels
            labels  : names of high-symmetry points
        """
        Nk = geometry.N_k
        b1, b2 = geometry.b1, geometry.b2
        Gamma = (0.0, 0.0)
        M     = (0.5 * b1).tolist()
        K     = ((2*b1 + b2) / 3).tolist()
        path = [("G", Gamma), ("M", M), ("K", K), ("G", Gamma)]

        kpoints, dist, ticks, labels = [], [0.0], [], []
        cumd = 0.0

        for i in range(len(path) - 1):
            label_i, k_i = path[i]
            label_j, k_j = path[i+1]
            ticks.append(cumd)
            labels.append(label_i)
            for t in range(Nk):
                frac = t / Nk
                kx = k_i[0] + frac * (k_j[0] - k_i[0])
                ky = k_i[1] + frac * (k_j[1] - k_i[1])
                if kpoints:
                    dk = np.hypot(kx - kpoints[-1][0], ky - kpoints[-1][1])
                    cumd += dk
                kpoints.append((kx, ky))
                dist.append(cumd)

        ticks.append(cumd)
        labels.append(path[-1][0])
        # drop last dist so that len(dist) == len(kpoints)
        return kpoints, dist[:-1], ticks, labels

    def _reshape_Ek_into_grid(self, kx_grid, ky_grid):
        """
        Internal helper to reshape E_k_dict into a 3D array E_3d[ix, iy, band].
        """
        n_kx, n_ky = len(kx_grid), len(ky_grid)
        first = next(iter(self.E_k_dict))
        N_bands = self.E_k_dict[first].shape[0]
        E_3d = np.zeros((n_kx, n_ky, N_bands))
        for ix, kx in enumerate(kx_grid):
            for iy, ky in enumerate(ky_grid):
                E_3d[ix, iy, :] = self.E_k_dict[f"[{kx}, {ky}]"]
        return E_3d

    def plot_band_structure(self, geometry:Geometry, bands:list = [], 
                            hide:bool=True):
        """
        Plot all bands whose energies are non-zero at all k-points.
        """
        N_bands = len(self.sublattice_idxs) * len(self.coupled_states)
        band_dict = self.band_structure_data["band_dict"]
        path = self.band_structure_data["path"]
        ticks = self.band_structure_data["ticks"]
        labels = self.band_structure_data["labels"]
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0, 1, N_bands))
        if bands == []:
            bands = [i for i in range(N_bands)]

        fig, ax = plt.subplots(figsize=(8, 5))
        for idx, energies in band_dict.items():
            if idx not in bands:
                continue
            if hide and np.all(np.abs(energies) < 1e-8):
                continue
            ax.plot(path, energies, lw=1.5, color=colors[idx])

        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(path[0], path[-1])
        ax.set_xlabel("k-path", fontsize=12)
        ax.set_ylabel("E/eV", fontsize=12)
        ax.grid(True, ls="--", lw=0.5)
        plt.tight_layout()
        plt.show()
    
    def plot_dispersion(self, geometry: Geometry, legend:bool=False, 
                        hide:bool=True):  
        kx, ky = geometry.kx_bulk, geometry.ky_bulk
        n_kx, n_ky = len(kx), len(ky)
        E_k_list = []
        for k_x in kx:
            for k_y in ky:
                key = f"[{k_x}, {k_y}]"
                E_k_list.append(self.E_k_dict[key])
        E_stacked = np.stack(E_k_list)  # Shape: (n_kx * n_ky, n_bands)
        E_3d = E_stacked.reshape(n_kx, n_ky, -1)
        n_bands = E_3d.shape[2]  # nº eigenvalues per k-point
        KX, KY = np.meshgrid(kx, ky, indexing='ij') 
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for band in range(n_bands):
            E = E_3d[:, :, band]
            if np.allclose(E, 0, rtol=1e-6):# and hide:
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