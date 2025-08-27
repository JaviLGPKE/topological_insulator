import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, cKDTree

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
        self.lattice_constant = parser.lattice_constant.value
        self.buckling_cosine = parser.buckling_cosine.value
        self.lattice_vectors = lattice_vectors =  parser.lattice_vectors.value
        self.a1, self.a2 = np.array(lattice_vectors[0]), np.array(lattice_vectors[1])
        self.delta_vectors =  np.array(parser.delta_vectors.value)
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
        self.N_k = N_k = self.model_options.N_k
        self.dangling_bonds = dangling_bonds = self.model_options.dangling_bonds
        assert(len(self.lattice_vectors[0]) == self.n_dim)
        print(f"Building Geometry...")
        self._build_lattice(N_r)
        self._set_connectivity_NN()
        if not dangling_bonds:
            # NOTE: do not move position of logic
            self._prune_dangling()
        self._set_connectivity_NNN()
        self._build_brillouine_zone(N_k)
        print(f"Geometry - Done.")

    def _build_lattice(self, N_r, dangling_bonds:bool=True):
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
                if abs(dist - 1) < tol:
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
            # Enforce Symmetry
            for k in nnn_candidates:
                C[i, k] = 1
                C[k, i] = 1
        self.nnn_connectivity_matrix = C

    def _prune_dangling(self):
        """
        Remove dangling bonds and update edge sites based on coordination numbers.
        """
        edge_set = set()
        for site in self.edge_sites:
            rounded_site = (round(site[0], 8), round(site[1], 8))
            edge_set.add(rounded_site)

        is_edge = [
            (round(site[0], 8), round(site[1], 8)) in edge_set
            for site in self.sites
        ]
        coordinations = [len(nn) for nn in self.nn_list]
        distinct_labels = np.unique(self.sublattice_label_idxs)
        typical_nn_dict = {}
        for label in distinct_labels:
            non_edge_indices = [
                idx for idx in range(len(self.sites))
                if (self.sublattice_label_idxs[idx] == label) and (not is_edge[idx])
            ]
            if not non_edge_indices:
                typical_nn_dict[label] = None
            else:
                bulk_coords = [coordinations[i] for i in non_edge_indices]
                values, counts = np.unique(bulk_coords, return_counts=True)
                typical_nn_dict[label] = values[np.argmax(counts)]
        keep_indices = []
        for idx in range(len(self.sites)):
            if not is_edge[idx]:
                keep_indices.append(idx)
            else:
                label = self.sublattice_label_idxs[idx]
                typical_nn = typical_nn_dict[label]
                if typical_nn is None:
                    keep_indices.append(idx)
                elif coordinations[idx] >= typical_nn - 1:
                    keep_indices.append(idx)
        # Update lattice properties
        keep_indices = np.array(keep_indices)
        self.sites = self.sites[keep_indices]
        self.sublattice_label_idxs = self.sublattice_label_idxs[keep_indices]
        # Rebuild connectivity with remaining sites
        self._set_connectivity_NN()
        # Update edge sites
        self._update_edge_sites()

    def _update_edge_sites(self):
        sites = self.sites
        pts = sites[:, :2]
        tri = Delaunay(pts)
        edges = {}
        for simplex in tri.simplices:
            for i,j in ((0,1),(1,2),(2,0)):
                a, b = sorted([simplex[i], simplex[j]])
                edges.setdefault((a,b), 0)
                edges[(a,b)] += 1
        boundary_edges = [edge for edge,count in edges.items() if count == 1]
        edge_indices = sorted({i for e in boundary_edges for i in e})
        tree  = cKDTree(sites)
        seed_idxs = np.array(edge_indices, dtype=int)
        radius = 1.01
        bonded = tree.query_ball_point(sites[seed_idxs], r=radius)
        all_edge = set(seed_idxs.tolist())
        for nbr_list in bonded:
            all_edge.update(nbr_list)
        self.edge_indices = sorted(all_edge)
        self.edge_sites   = sites[self.edge_indices]
        
    def get_edge_from_sites(self):
        site_dict = {}
        for idx, site in enumerate(self.sites):
            key = (round(site[0], 8), round(site[1], 8))
            site_dict[key] = idx
        
        indices = []
        for coord in self.edge_sites:
            key = (round(coord[0], 8), round(coord[1], 8))
            if key in site_dict:
                indices.append(site_dict[key])
        return indices

    def get_label(self, idx):
        return self.sublattice_labels[self.sublattice_label_idxs[idx]]

    def _build_brillouine_zone(self, N_k):
        factor = 2
        a1, a2 = self.a1, self.a2
        A = a1[0]*a2[1] - a1[1]*a2[0]
        b1 = self.b1 = (2*np.pi/A) * np.array([a2[1], -a2[0]])
        b2 = self.b2 = (2*np.pi/A) * np.array([-a1[1], a1[0]])
        K_point = self.K_point = ((2*b1 + b2)/3).tolist()
        Gamma = self.Gamma = [0.0, 0.0]
        trims = self.trims = [Gamma, 0.5*b1, 0.5*b2, 0.5*(b1+b2)]
        # Bulk
        if self.model_options.BZ == "reduced":
            discretization = np.linspace(-np.pi, np.pi, N_k)
        elif self.model_options.BZ == "extended":
            discretization = np.linspace(-factor*np.pi, factor*np.pi, N_k)
        else:
            raise NotImplementedError(f"'{self.model_options.BZ}' Not Implemented!")

        all_points = discretization.tolist()
        for point in trims + [K_point]:
            all_points.extend(point)  # Add both x and y
        k_common = np.unique(all_points)
        self.kx_bulk = self.ky_bulk = k_common
        self.N_k = len(k_common)
        self.kx_bulk = self.ky_bulk = k_common
        self.N_k = len(k_common)
        self.kx_grid, self.ky_grid = np.meshgrid(k_common, k_common, indexing='xy')
        self.BZ_mask = self.brillouin_zone_mask(
                k_common, k_common, b1, b2, M=2, tol=1e-12)
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
    
    def brillouin_zone_mask(self, kx, ky, b1, b2, M=2, tol=1e-12):
        """
        Wigner-Seitz construction of the first BZ i.e.
        the set of k closer to Gamma(0,0) than any other reciprocal vector.

        returns: boolean mask shape=(len(kx), len(ky)) -> True if k in first BZ.
        """
        N_kx = len(kx); N_ky = len(ky)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.stack([KX.ravel(), KY.ravel()], axis=1)
        # build small set of reciprocal lattice points R = m*b1 + n*b2
        ms = np.arange(-M, M+1)
        ns = np.arange(-M, M+1)
        R_list = []
        for m in ms:
            for n in ns:
                R_list.append(m * np.asarray(b1) + n * np.asarray(b2))
        R = np.vstack(R_list)  # (numR, 2)
        zero_idx = np.all(np.abs(R) < 1e-12, axis=1)
        displacements = R[~zero_idx] # don't compare with itself
        dist0 = np.sum(K**2, axis=1)
        distR = np.sum((K[:, None, :] - displacements[None, :, :])**2, axis=2)
        # inside BZ iff dist0 <= all distR
        inside_flat = np.all(dist0[:, None] <= distR + tol, axis=1)
        mask = inside_flat.reshape(N_kx, N_ky)
        return mask

    def get_location_idx(self, location:str):
        sites = self.sites
        x_max, y_max = max(sites[:, 0]), max(sites[:, 1])
        x_min, y_min = min(sites[:, 0]), min(sites[:, 1])
        if location == "bulk":
            x_idxs = np.where(np.isclose(sites[:, 0], x_max/2, rtol=2e-1))[0]
            y_idxs = np.where(np.isclose(sites[:, 1], y_max/2, rtol=2e-1))[0]
            idx_candidates = np.intersect1d(x_idxs, y_idxs)
        elif location == "edge":
            edge_sites = self.edge_sites
            x_idxs = np.where(np.isclose(edge_sites[:, 0], x_max/3, rtol=5e-1))[0]
            y_idxs = np.where(np.isclose(edge_sites[:, 1], y_min*0.90, rtol=5e-1))[0]
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

    def _find_unit_cell(self, sub_A_idx, atol=1e-8):
        unit_cell = [sub_A_idx]
        for n, d in enumerate(self.delta_vectors):
            if n == 0:
                # 1st delta vector corresponds to [0, 0], equivalent to sublattice A
                continue 
            site = self.sites[sub_A_idx].copy() + d
            idx = np.where(np.all(np.isclose(self.sites, site, atol=atol), axis=1))[0][0]
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

    def get_chirality(self, idx_i, idx_j):
        neighbours_i = self.get_neighbour_idxs(idx_i)
        neighbours_j = self.get_neighbour_idxs(idx_j)
        shared_neighbors = set(neighbours_i).intersection(neighbours_j)
        if not shared_neighbors:
            raise ValueError(f"No shared neighbor between {idx_i} and {idx_j}")
        idx_k = next(iter(shared_neighbors))  # Take the first shared neighbor
        r_i = np.array(self.sites[idx_i])
        r_j = np.array(self.sites[idx_j])
        r_k = np.array(self.sites[idx_k])
        d1 = r_k - r_i
        d2 = r_j - r_k
        nu_ij = d1[0] * d2[1] - d1[1] * d2[0]
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
    
    def get_phase_idxs(self, idx_i:int, dm_dict:dict, sublattice_idxs:list):
        phase_dict = {}
        unit_cell_idxs = [idx for idx in dm_dict.keys() if idx in sublattice_idxs]
        non_unit_cell_idxs = [idx for idx in dm_dict.keys() if idx not in sublattice_idxs]
        for idx_j, m_ij in dm_dict.items():
            if idx_j in non_unit_cell_idxs:
                idx_j_phase = idx_j
                idx_j = self._find_site(idx_j_phase, m_ij, sublattice_idxs)
                if idx_j is None:
                    continue # skip phases that don't have associated indexes
                phase_dict[idx_j] = idx_j_phase
            elif idx_j in unit_cell_idxs:
                if idx_j in phase_dict.keys():
                    continue # skip idx that has established phase
                phase_dict[idx_j] = None
            else:
                raise ValueError(f"'{idx_j}' not in dm_dict")
        return phase_dict

    def _find_site(self, idx_j_phase, m_ij, sublattice_idxs):
        T = self.T
        phase_site = self.sites[idx_j_phase].copy()
        if m_ij > 0: # positive direction
            phase_site += T
        else: # negative direction
            phase_site -= T
        idx_j = np.where(
            np.all(np.isclose(self.sites, phase_site, atol=1e-8), axis=1))[0][0]
        if idx_j in sublattice_idxs:
            return idx_j
        else:
            return None # Bond offers no contribution?

    def plot_lattice(self, sites_of_interest=None, ax=None):
        """
        Plots the 2D geometry of the lattice:
        - Sites as colored dots (each color = one sublattice).
        - Bonds/edges where NN connectivity_matrix[i,j] == 1.

        Parameters
        ----------
        sites_of_interest: allows user to highlight sites using 
            the idxs corresponding indexes
        ax : matplotlib.axes._axes.Axes, optional
            If provided, the function will draw on this Axes.
            Otherwise, it will create a new figure and Axes.
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
