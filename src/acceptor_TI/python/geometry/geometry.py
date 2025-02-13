import numpy as np
from matplotlib import pyplot as plt

from ..model_options import ModelOptions
from ..cell_parser import CellParser

class Geometry:
    def __init__(self, model_options:ModelOptions, cell_parser:CellParser):
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.name = cell_parser.general.name.value
        self.n_dim = cell_parser.general.dimensions.value
        self.n_sublattices = len(cell_parser.geometry.delta_vectors.value)
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]

    def build_lattice(self, size, N_k) -> None:
        """
        Builds the lattice structure in real space.

        Parameters
        ----------
        parser: Parameter
            The dictionary containing the geometrical parameters to replicate the lattice.
        """
        self.N_k = N_k
        parser = self.cell_parser.geometry
        lattice_vectors = parser.lattice_vectors.value
        assert(len(lattice_vectors[0]) == self.n_dim)

        print(f"Building Geometry...")
        self._build_lattice(size, parser)
        self._set_connectivity(parser)
        self._build_brillouine_zone(size, N_k)
        print(f"Geometry Completed.")

    def _build_lattice(self, size, parser):
        self.lattice_constant = a = parser.lattice_constant.value
        lattice_vectors = parser.lattice_vectors.value
        self.a1, self.a2 = a1, a2 = lattice_vectors[0], lattice_vectors[1]
        delta_vectors = parser.delta_vectors.value
        # Build lattice
        sites, sublattice_label = [], []
        site_index = 0
        for i in range(size):
            for j in range(size):
                for s, delta in enumerate(delta_vectors):
                    x = (i * a1[0] + j * a2[0] + delta[0]) * a
                    y = (i * a1[1] + j * a2[1] + delta[1]) * a
                    sites.append([x, y])
                    sublattice_label.append(s)
                    site_index += 1
        self.sites = np.array(sites)
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

    def _build_brillouine_zone(self, size, N_k):
        # Reciprocal vectors
        a = self.lattice_constant
        a1, a2 = self.a1, self.a2
        area = a1[0]*a2[1] - a1[1]*a2[0]
        self.b1 = b1 = (2*np.pi / area) * np.array([a2[1], -a2[0]])
        self.b2 = b2 = (2*np.pi / area) * np.array([-a1[1], a1[0]])
        # Generate grid for 3D dispersion plot
        if self.model_options.dispersion:
            self.kx_vals, self.ky_vals = kx_vals, ky_vals = (
                np.linspace(-np.pi/a, np.pi/a, N_k), np.linspace(-np.pi/a, np.pi/a, N_k))
            self.kx_grid, self.ky_grid = np.meshgrid(kx_vals, ky_vals)
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
            #     for t in np.linspace(0, 1, size**2):
            #         frac_coords = (1 - t)*start + t*end
            #         k = frac_coords[0]*b1 + frac_coords[1]*b2
            #         self.k_path.append(k)
            # self.k_path = np.array(self.k_path)
    
    def get_bulk_idx(self):
        a = self.lattice_constant
        bulk_sublattices = [0]
        sites = self.sites
        x_max = max(sites[:, 0])
        y_max = max(sites[:, 1])
        bulk_x_idxs = np.where(np.isclose(sites[:, 0], x_max/2, rtol=1e-1*a))[0]
        bulk_y_idxs = np.where(np.isclose(sites[:, 1], y_max/2, rtol=1e-1*a))[0]
        bulk_idx_candidates = np.intersect1d(bulk_x_idxs, bulk_y_idxs)
        chosen_bulk = [c for c in bulk_idx_candidates if self.sublattice_label_idxs[c] in bulk_sublattices]
        if not chosen_bulk:
            raise ValueError(f"No site found near the center in sublattices = {bulk_sublattices}!")
        bulk_idx = chosen_bulk[0]
        return bulk_idx

    def get_neighbours_data(self, bulk_idx):
        C = self.connectivity_matrix
        neighbours_idx = np.where(C[bulk_idx, :] == 1)[0]
        return neighbours_idx

    def get_dr(self, bulk_idx, neighbour_idxs, type="list"):
        if type == "list":
            return [self.sites[n] - self.sites[bulk_idx] for n in neighbour_idxs]
        elif type == "dict":
            return {n: self.sites[n] - self.sites[bulk_idx] for n in neighbour_idxs}

    def get_edge_idx(self):
        # TODO: similar process to bulk but for the continuous edges
        return

    def bond_orientation(self, dr_list):
        cos_theta_list = []
        for dr in dr_list:
            bond_length = np.linalg.norm(dr)
            assert(bond_length != 0)
            cos_theta = dr / bond_length
            cos_theta_list.append(cos_theta)
        return np.array(cos_theta_list)

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
