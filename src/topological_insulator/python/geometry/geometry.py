import numpy as np
from matplotlib import pyplot as plt

from ..model_options import ModelOptions
from ..cell_parser import CellParser

from IPython import embed
class Geometry:
    def __init__(self, model_options:ModelOptions, cell_parser:CellParser):
        # Setup
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.name = cell_parser.general.name.value
        self.n_dim = cell_parser.general.dimensions.value
        self.n_sublattices = len(cell_parser.geometry.delta_vectors.value)
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        self.label_mapper = {idx: label for idx, label in enumerate(self.sublattice_labels)}
        self.idx_mapper = {label: idx for idx, label in enumerate(self.sublattice_labels)}
        # Vectors
        parser = self.cell_parser.geometry
        self.lattice_constant = a = parser.lattice_constant.value
        lattice_vectors = parser.lattice_vectors.value
        self.a1, self.a2 = a * np.array(lattice_vectors[0]), a * np.array(lattice_vectors[1])
        self.delta_vectors = a * np.array(parser.delta_vectors.value)
        for n, d in enumerate(self.delta_vectors):
            setattr(self, f"d_{n+1}", np.array(d))

    def build_lattice(self) -> None:
        """
        Builds the lattice structure in real space.

        Parameters
        ----------
        parser: Parameter
            The dictionary containing the geometrical parameters to replicate the lattice.
        """
        self.N_r = N_r = self.model_options.N_r
        self.N_k = self.model_options.N_k
        self.dangling_bonds = dangling_bonds = self.model_options.dangling_bonds

        parser = self.cell_parser.geometry
        lattice_vectors = parser.lattice_vectors.value
        assert(len(lattice_vectors[0]) == self.n_dim)

        print(f"Building Geometry...")
        self._build_lattice(N_r)
        self._set_connectivity_NN()
        self._set_connectivity_NNN()
        self._build_brillouine_zone()
        print(f"Geometry - Done.")

    def _build_lattice(self, N_r):
        a1, a2 = self.a1, self.a2
        # Build lattice
        sites, edge_sites, sublattice_label = [], [], []
        site_index = 0
        for i in range(N_r):
            for j in range(N_r):
                for s, d in enumerate(self.delta_vectors):
                    x = (i * a1[0] + j * a2[0] + d[0])
                    y = (i * a1[1] + j * a2[1] + d[1])
                    site = (x, y)
                    if i == 0 or i == N_r - 1 or j == 0 or j == N_r - 1:
                        edge_sites.append(site)
                    sites.append(site)
                    sublattice_label.append(s)
                    site_index += 1
        self.sites, self.edge_sites = np.array(sites), np.array(edge_sites)
        self.sublattice_label_idxs = np.array(sublattice_label, dtype=int)
        self.distinct_labels = np.unique(self.sublattice_label_idxs[self.sublattice_label_idxs != 0])

    def _set_connectivity_NN(self, tol=1e-12) -> None:
        """
        Sets the connectivity matrix based on whether the distance between two sites
        is within 'reference_dist'.

        Parameters
        ----------
        reference_dist : float
            The distance at which two sites are considered connected.
        tol : float
            Tolerance for considering distances as equal to 'reference_dist'.
        """
        sites = self.sites
        a = self.lattice_constant
        N = len(sites)     
        C = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(i + 1, N):
                # Sum of squared distances
                dist_sq = 0.0
                for d in range(self.n_dim):
                    diff = sites[i][d] - sites[j][d]
                    dist_sq += diff * diff
                dist = np.sqrt(dist_sq)
                # Nearest Neighbours
                if abs(dist - a) < tol:
                    C[i, j] = 1
                    C[j, i] = 1 # h.c.
        self.nn_connectivity_matrix = C
        # Build nn_list: list of nearest neighbors for each site
        nn_list = [[] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if C[i, j] == 1:
                    nn_list[i].append(j)
        self.nn_list = nn_list
    
    def _set_connectivity_NNN(self) -> None:
        """
        Sets the NNN connectivity matrix based on NN connectivity.
        Two sites are NNN if they share a common NN neighbor.
        """
        N = len(self.sites)
        C = np.zeros((N, N), dtype=int)
        nn_list = self.nn_list
        for i in range(N):
            neighbors_of_i = set(nn_list[i])
            nnn_candidates = set()
            for j in nn_list[i]:
                for k in nn_list[j]:
                    if k == i:
                        continue
                    if k not in neighbors_of_i: # Exclude direct neighbors
                        nnn_candidates.add(k)
            # Set symmetric connections
            for k in nnn_candidates:
                C[i, k] = 1
                C[k, i] = 1
        self.nnn_connectivity_matrix = C

    def get_label(self, idx):
        return self.sublattice_labels[self.sublattice_label_idxs[idx]]

    def _build_brillouine_zone(self):
        factor = 2
        N_k = self.N_k
        a = self.lattice_constant
        a1, a2 = self.a1, self.a2
        A = a1[0]*a2[1] - a1[1]*a2[0]
        b1 = self.b1 = (2*np.pi/A) * np.array([a2[1], -a2[0]])
        b2 = self.b2 = (2*np.pi/A) * np.array([-a1[1], a1[0]])
        trims = self.trims = [np.array([0.0, 0.0]), 0.5*b1, 0.5*b2, 0.5*(b1+b2)]
        # Bulk
        factor = 2
        if self.model_options.BZ == "reduced":
            discretization = np.linspace(-np.pi/a, np.pi/a, N_k)
        elif self.model_options.BZ == "extended":
            discretization = np.linspace(-factor*np.pi/a, factor*np.pi/a, N_k)
        else:
            raise NotImplementedError(f"'{self.model_options.BZ}' Not Implemented!")
        # Include trim points in k-space
        trim_kx = [t[0] for t in trims]
        trim_ky = [t[1] for t in trims]
        kx_bulk = np.unique(np.concatenate([discretization, trim_kx]))
        ky_bulk = np.unique(np.concatenate([discretization, trim_ky]))
        self.kx_bulk, self.ky_bulk = kx_bulk, ky_bulk
        self.kx_grid, self.ky_grid = np.meshgrid(kx_bulk, ky_bulk)
        # Edge
        if self.model_options.location in ["edge", "both"]:
            T = a1 if a2[1] > a1[1] else a2
            self.T = T
            self.T_norm = T_norm = np.linalg.norm(T)
            self.T_hat = T/T_norm
            if self.model_options.BZ == "reduced":
                discretization_edge = np.linspace(-np.pi/(T_norm), np.pi/(T_norm), N_k)
            elif self.model_options.BZ == "extended":
                discretization_edge = np.linspace(-factor*np.pi/(T_norm), factor*np.pi/(T_norm), N_k)
            else:
                raise NotImplementedError(f"'{self.model_options.BZ}' Not Implemented!")
            self.k_edge = discretization_edge

    def get_location_idx(self, location:str):
        a = self.lattice_constant
        sites = self.sites
        x_max, y_max = max(sites[:, 0]), max(sites[:, 1])
        x_min, y_min = min(sites[:, 0]), min(sites[:, 1])
        if location == "bulk":
            x_idxs = np.where(np.isclose(sites[:, 0], x_max/2, rtol=2e-1*a))[0]
            y_idxs = np.where(np.isclose(sites[:, 1], y_max/2, rtol=2e-1*a))[0]
            idx_candidates = np.intersect1d(x_idxs, y_idxs)
        elif location == "edge":
            edge_sites = self.edge_sites
            x_idxs = np.where(np.isclose(edge_sites[:, 0], x_max/3, rtol=2.2e-1*a))[0]
            y_idxs = np.where(np.isclose(edge_sites[:, 1], y_min*0.90, rtol=2.5e-1*a))[0]
            edge_idxs = np.intersect1d(x_idxs, y_idxs)
            candidate_edge_sites = edge_sites[edge_idxs]
            idx_candidates = [np.where((sites == candidate).all(axis=1))[0][0] for candidate in candidate_edge_sites]
        else: 
            raise ValueError(f"Location '{location}' not available")
        chosen_idxs = [c for c in idx_candidates if self.sublattice_label_idxs[c] == 0]
        return chosen_idxs[0]

    def get_sublattice_idxs(self, location: str):
        chosen_idx = self.get_location_idx(location)
        setattr(self, f"{location}_idx", chosen_idx)
        unit_cell_idxs = self._find_unit_cell(chosen_idx)
        return sorted(unit_cell_idxs, key=lambda idx: self.sublattice_label_idxs[idx])

    def _find_unit_cell(self, sub_A_idx, tol=1e-5):
        unit_cell = [sub_A_idx]
        for n, d in enumerate(self.delta_vectors):
            if n == 0:
                # 1st delta vector corresponds to [0, 0], equivalent to sublattice A
                continue 
            site = self.sites[sub_A_idx].copy() + d
            idx = np.where(np.all(np.isclose(self.sites, site, atol=1e-8), axis=1))[0][0]
            unit_cell.append(idx)
        assert(len(unit_cell) == self.n_sublattices)
        return unit_cell

    def get_neighbour_idxs(self, site_idx):
        C = self.nn_connectivity_matrix
        neighbours_idx = np.where(C[site_idx, :] == 1)[0]
        return neighbours_idx

    def get_next_neighbour_idxs(self, site_idx):
        C = self.nnn_connectivity_matrix
        next_neighbours_idx = np.where(C[site_idx, :] == 1)[0]
        return next_neighbours_idx

    def get_chirality(self, site_i, site_j):
        neighbours_i = self.get_neighbour_idxs(site_i)
        neighbours_j = self.get_neighbour_idxs(site_j)
        shared_neighbors = set(neighbours_i).intersection(neighbours_j)
        if not shared_neighbors:
            raise ValueError(f"No shared neighbor between {site_i} and {site_j}")
        k = next(iter(shared_neighbors))  # Take the first shared neighbor
        r_i = np.array(self.sites[site_i])
        r_j = np.array(self.sites[site_j])
        r_k = np.array(self.sites[k])
        d1 = r_k - r_i
        d2 = r_j - r_k
        cross_z = d1[0] * d2[1] - d1[1] * d2[0]
        nu_ij = int(np.sign(cross_z))
        return nu_ij

    def get_dr(self, location, bulk_idx, neighbour_idxs, type="list"):
        dr_list, dm_list = [], []
        i = bulk_idx
        for j in neighbour_idxs:
            r_ij = self.sites[i] - self.sites[j]
            dr_list.append(r_ij)
            dm_list.append(np.dot(r_ij, self.T_hat) if location == "edge" else 0)
        if type == "dict":
            dr_dict = {n: dr_list[i] for i, n in enumerate(neighbour_idxs)}
            dm_dict = {n: dm_list[i] for i, n in enumerate(neighbour_idxs)}
            return dr_dict, dm_dict
        elif type == "list":
            return dr_list, dm_list

    def bond_orientation(self, dr_list):
        cosines_list = []
        for dr in dr_list:
            bond_length = np.linalg.norm(dr)
            assert(bond_length != 0)
            cosines = dr / bond_length
            cosines_list.append(cosines)
        return np.array(cosines_list)

    def get_edge_path(self, sublattices: list):
        sites = self.sites
        a1, a2 = self.a1, self.a2 
        # NOTE: we start from the bottom edge, so we need to go backwards
        # along the opposite direction of the descending basis vector
        a = a2 if self.a1[1] > self.a2[1] else a1
        sublattices_considered = {}
        for idx in sublattices:
            label = self.sublattice_label_idxs[idx]
            sublattices_considered[label] = []
            path = sites[idx].copy() 
            for _ in range(self.N_r-1):
                path -= a
                site_i = np.where(np.all(np.isclose(sites, path, atol=1e-8), axis=1))[0]
                if len(site_i) == 0:
                    raise ValueError(f"Site {path} not found in self.sites")
                sublattices_considered[label].append(site_i[0])
        return sublattices_considered
    
    def _get_phase_idxs(self, idx_i:int, dm_dict:dict, sublattice_idxs:list, term:str="NN"):
        phase_dict = {}
        unit_cell_idxs = [idx for idx in dm_dict.keys() if idx in sublattice_idxs]
        non_unit_cell_idxs = [idx for idx in dm_dict.keys() if idx not in sublattice_idxs]
        for idx_j, m_ij in dm_dict.items():
            if idx_j in non_unit_cell_idxs:
                idx_j_phase = idx_j
                find_phase = getattr(self, f"_find_phase_{term}")
                idx_j = find_phase(idx_j_phase, m_ij)
                phase_dict[idx_j] = idx_j_phase
            elif idx_j in unit_cell_idxs:
                if idx_j in phase_dict.keys():
                    # FIXME: lists to append multiple phases?
                    # skip idx that has established phase
                    continue
                phase_dict[idx_j] = None
            else:
                raise ValueError(f"'{idx_j}' not in dm_dict")
        return phase_dict
    
    def _find_phase_NN(self, idx_j, m_ij):
        T = self.T
        phase_site = self.sites[idx_j].copy()
        if m_ij > 0: # left direction
            phase_site += T
        else: # right direction
            phase_site -= T
        idx_j_phase = np.where(
            np.all(np.isclose(self.sites, phase_site, atol=1e-8), axis=1))[0][0]
        return idx_j_phase

    def _find_phase_NNN(self, idx_j, m_ij):
        embed()
        phase_site = self.sites[idx_j].copy()
        return

    def plot_lattice(self, ax=None, sites_of_interest=None):
        """
        Plots the 2D geometry of the lattice:
        - Sites as colored dots (each color = one sublattice).
        - Bonds/edges where NN connectivity_matrix[i,j] == 1.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            If provided, the function will draw on this Axes.
            Otherwise, it will create a new figure and Axes.
        sites_of_interest: allows user to highlight sites using 
            the idxs corresponding indexes
        """
        if self.n_dim != 2:
            raise ValueError("plot_geometry is designed for 2D lattices (n_dim=2).")
        sites = self.sites
        sublat_full = self.sublattice_label_idxs
        C = self.nn_connectivity_matrix
        N = len(sites)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            new_figure_created = True
        else:
            new_figure_created = False
        # 1) Draw edges first so that the site markers overlay them
        for i in range(N):
            for j in range(i+1, N):
                if C[i, j] == 1:
                    x_i, y_i = sites[i]
                    x_j, y_j = sites[j]
                    ax.plot([x_i, x_j], [y_i, y_j], color="k", linewidth=0.5, alpha=0.6, zorder=1)
        # 2) Scatter plot of sites by sublattice
        color_list = color_list = ["yellow", "tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]
        unique_sublattices = np.unique(sublat_full)
        for s in unique_sublattices:
            mask = (sublat_full == s)
            label_str = self.sublattice_labels[s] if s < len(self.sublattice_labels) else f"Sublatt. {s}"
            ax.scatter(sites[mask, 0],
                    sites[mask, 1],
                    color=color_list[s % len(color_list)],
                    label=label_str,
                    s=20, alpha=0.9, zorder=2)
        # 3) Highlight sites_of_interest if provided
        if sites_of_interest is not None:
            sites_of_interest = np.asarray(sites_of_interest)
            if sites_of_interest.size > 0:
                if (sites_of_interest.dtype.kind not in ('i', 'u') or 
                    np.any(sites_of_interest < 0) or 
                    np.any(sites_of_interest >= N)):
                    raise ValueError("All elements in sites_of_interest must be integers within [0, N-1].")
                highlight_coords = sites[sites_of_interest]
                ax.scatter(
                    highlight_coords[:, 0], highlight_coords[:, 1],
                    color='black', s=40, edgecolors='black', linewidths=0.8,
                    zorder=3, label = "SoI"
                )
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Lattice geometry: {self.name}")
        ax.legend()
        if new_figure_created:
            plt.tight_layout()
            plt.show()
