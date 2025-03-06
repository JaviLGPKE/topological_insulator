import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

from ..model_options import ModelOptions
from ..cell_parser import CellParser

from IPython import embed
class Geometry:
    def __init__(self, model_options:ModelOptions, cell_parser:CellParser):
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.name = cell_parser.general.name.value
        self.n_dim = cell_parser.general.dimensions.value
        self.n_sublattices = len(cell_parser.geometry.delta_vectors.value)
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]

    def build_lattice(self) -> None:
        """
        Builds the lattice structure in real space.

        Parameters
        ----------
        parser: Parameter
            The dictionary containing the geometrical parameters to replicate the lattice.
        """
        self.N_r = N_r = self.model_options.N_r
        self.N_k = N_k = self.model_options.N_k
        parser = self.cell_parser.geometry
        lattice_vectors = parser.lattice_vectors.value
        assert(len(lattice_vectors[0]) == self.n_dim)

        print(f"Building Geometry...")
        self._build_lattice(N_r, parser)
        self._set_connectivity(parser)
        self.convex_hull = ConvexHull(self.sites)
        print(f"Geometry - Done.")

    def _build_lattice(self, N_r, parser):
        self.lattice_constant = a = parser.lattice_constant.value
        lattice_vectors = parser.lattice_vectors.value
        self.a1, self.a2 = a1, a2 = (np.array(lattice_vectors[0]), 
                                     np.array(lattice_vectors[1]))
        delta_vectors = parser.delta_vectors.value
        # Build lattice
        sites, edge_sites, sublattice_label = [], [], []
        site_index = 0
        for i in range(N_r):
            for j in range(N_r):
                for s, delta in enumerate(delta_vectors):
                    x = (i * a1[0] + j * a2[0] + delta[0]) * a
                    y = (i * a1[1] + j * a2[1] + delta[1]) * a
                    site = [x, y]
                    sites.append(site)
                    if i == 0 or i == N_r - 1 or j == 0 or j == N_r - 1:
                        edge_sites.append(site)
                    sublattice_label.append(s)
                    site_index += 1
        self.sites, self.edge_sites = np.array(sites), np.array(edge_sites)
        self.sublattice_label_idxs = np.array(sublattice_label, dtype=int)
        self.distinct_labels = np.unique(self.sublattice_label_idxs[self.sublattice_label_idxs != 0])

    def _set_connectivity(self, parser, tol=1e-12) -> None:
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
        a = parser.lattice_constant.value
        nn_factor = parser.lattice_constant.nn_factor
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
                if abs(dist - (nn_factor * a)) < tol:
                    C[i, j] = 1
                    C[j, i] = 1
        self.connectivity_matrix = C

    def _build_brillouine_zone(self, edge_idxs=None):
        # Reciprocal vectors
        N_k = self.N_k
        a = self.lattice_constant
        a1, a2 = self.a1, self.a2
        area = a1[0]*a2[1] - a1[1]*a2[0]
        self.b1 = b1 = (2*np.pi / area) * np.array([a2[1], -a2[0]])
        self.b2 = b2 = (2*np.pi / area) * np.array([-a1[1], a1[0]])
        self.kx_bulk, self.ky_bulk = kx_bulk, ky_bulk = (
            np.linspace(-np.pi/a, np.pi/a, N_k), np.linspace(-np.pi/a, np.pi/a, N_k))
        self.kx_grid, self.ky_grid = np.meshgrid(kx_bulk, ky_bulk)
        if self.model_options.location == "edge":
            # FIXME: this method is not general for any lattice
            self.T = T = a1 if self.a1[1] > self.a2[1] else a2
            self.T_norm = T_norm = np.linalg.norm(T)
            self.T_hat = T/T_norm
            k_max, k_min = np.pi/(a*T_norm), -1*np.pi/(a*T_norm) 
            self.k_edge = np.linspace(k_min, k_max, N_k)

        # TODO: Generate high-symmetry path for band structure
        # if self.model_options.band_structure:
            # gamma = np.array([0.0, 0.0])
            # k_point = np.array([1/3, 1/3])
            # m_point = np.array([0.5, 0.0])
            # path = [gamma, k_point, m_point, gamma]
            # self.k_path = []
            # for i in range(len(path)-1):
            #     start = path[i]
            #     end = path[i+1]
            #     for r in np.linspace(0, 1, N_r**2):
            #         frac_coords = (1 - r)*start + r*end
            #         k = frac_coords[0]*b1 + frac_coords[1]*b2
            #         self.k_path.append(k)
            # self.k_path = np.array(self.k_path)

    def get_location_idx(self, location:str):
        a = self.lattice_constant
        sublattices = [0]
        sites = self.sites
        x_max = max(sites[:, 0])
        y_max = max(sites[:, 1])
        if location == "bulk":
            x_idxs = np.where(np.isclose(sites[:, 0], x_max/2, rtol=1e-1*a))[0]
            y_idxs = np.where(np.isclose(sites[:, 1], y_max/2, rtol=1e-1*a))[0]
            idx_candidates = np.intersect1d(x_idxs, y_idxs)
        elif location == "edge":
            edge_sites = self.edge_sites
            edge_idxs = np.where(np.isclose(edge_sites[:, 0], 3*x_max/4, rtol=1e-1*a))[0]
            candidate_edge_sites = edge_sites[edge_idxs]
            idx_candidates = [np.where((sites == candidate).all(axis=1))[0][0] for candidate in candidate_edge_sites]
        else: 
            raise ValueError(f"Location '{location}' not available")
        chosen_idxs = [c for c in idx_candidates if self.sublattice_label_idxs[c] in sublattices]
        return chosen_idxs[0]

    def get_sublattice_idxs(self, location: str):
        # NOTE: must only consider unique sublattices that match
        # within coordinate conditions
        chosen_idx = self.get_location_idx(location)
        setattr(self, f"{location}_idx", chosen_idx)
        neighbour_idxs = self.get_neighbour_idxs(chosen_idx)
        sublattice_idxs = [chosen_idx]
        a = self.lattice_constant
        if location == "bulk":
            sublattice_idxs.extend(
                [idx for i, idx in enumerate(neighbour_idxs) if i < (self.n_sublattices - 1)]
            )
        elif location == "edge":
            edge_candidates, non_edge_candidates  = [], []
            for i, idx in enumerate(neighbour_idxs):
                site = self.sites[idx]
                if site in self.edge_sites:
                    edge_candidates.append(idx)
                else:
                    non_edge_candidates.append(idx)
            current_labels = {self.sublattice_label_idxs[i] for i in sublattice_idxs}
            # Add edge candidates if their sublattice label is new.
            for idx in edge_candidates:
                label = self.sublattice_label_idxs[idx]
                if label not in current_labels:
                    sublattice_idxs.append(idx)
                    current_labels.add(label)
                if len(sublattice_idxs) == self.n_sublattices:
                    break
            # If the unit cell is still incomplete, add from non-edge candidates.
            if len(sublattice_idxs) < self.n_sublattices:
                for idx in non_edge_candidates:
                    label = self.sublattice_label_idxs[idx]
                    if label not in current_labels:
                        sublattice_idxs.append(idx)
                        current_labels.add(label)
                    if len(sublattice_idxs) == self.n_sublattices:
                        break
        # Sort idxs according to label order
        sublattice_idxs = sorted(sublattice_idxs, key=lambda idx: self.sublattice_label_idxs[idx])
        return sublattice_idxs

    def get_neighbour_idxs(self, idx):
        C = self.connectivity_matrix
        neighbours_idx = np.where(C[idx, :] == 1)[0]
        return neighbours_idx

    def get_dr(self, location, bulk_idx, neighbour_idxs, type="list"):
        T_hat, T_norm = self.T_hat, self.T_norm
        dr_list, dm_list = [], []
        i = bulk_idx
        for j in neighbour_idxs:
            r_ij = self.sites[i] - self.sites[j]
            if location == "edge":
                projection = np.dot(r_ij, T_hat) / T_norm
                m_ij = projection  # translation count
                dm_list.append(m_ij)
                # print(bulk_idx, n, dm, r_ij, r_ij - (dm * self.T))
                r_ij -= m_ij * self.T  # unwrap displacement
            dr_list.append(r_ij)
        if type == "dict":
            dr_dict = {n: dr_list[i] for i, n in enumerate(neighbour_idxs)}
            dm_dict = {n: dm_list[i] for i, n in enumerate(neighbour_idxs)}
            return dr_dict, dm_dict
        elif type == "list":
            return dr_list, dm_list

    def bond_orientation(self, dr_list):
        cos_theta_list = []
        for dr in dr_list:
            bond_length = np.linalg.norm(dr)
            assert(bond_length != 0)
            cos_theta = dr / bond_length
            cos_theta_list.append(cos_theta)
        return np.array(cos_theta_list)

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

    def plot_lattice(self, ax=None):
        """
        Plots the 2D geometry of the lattice:
        - Sites as colored dots (each color = one sublattice).
        - Bonds/edges where connectivity_matrix[i,j] == 1.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            If provided, the function will draw on this Axes.
            Otherwise, it will create a new figure and Axes.
        """
        if self.n_dim != 2:
            raise ValueError("plot_geometry is designed for 2D lattices (n_dim=2).")
        sites = self.sites                         # shape (N, 2)
        sublat_full = self.sublattice_label_idxs   # shape (N,)
        C = self.connectivity_matrix               # shape (N, N)
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
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Lattice geometry: {self.name}")
        ax.legend()
        if new_figure_created:
            plt.tight_layout()
            plt.show()