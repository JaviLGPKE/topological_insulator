import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, cycle

from ..cell_parser import CellParser

class Geometry:
    def __init__(self, cell_parser:CellParser):
        self.name = cell_parser.general.name.value
        self.n_dim = cell_parser.general.dimensions.value
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        self.build_lattice(parser=cell_parser.geometry)

    def build_lattice(self, parser):
        """
        Builds the lattice structure in real space.

        Parameters
        ----------
        parser: Parameter
            The dictionary containing the geometrical parameters to replicate the lattice.
        """
        n_sub = parser.sublattices.value
        size = parser.size.value
        n_dim = self.n_dim
        a = parser.lattice_constant.value
        lattice_vectors = parser.lattice_vectors.value
        delta_vectors = parser.delta_vectors.value
        
        assert(n_sub == len(delta_vectors))
        assert(len(lattice_vectors[0]) == self.n_dim)

        # Build lattice
        coordinates, labels = [], []       
        for index_tuple in product(*(range(s) for s in size)): # e.g. (i, j) for 2D
            for s, delta in enumerate(delta_vectors):
                site = [0.0] * n_dim
                for k in range(n_dim):
                    for d in range(n_dim):
                        site[d] += a * (index_tuple[k] * lattice_vectors[k][d])
                for d in range(n_dim):
                    site[d] += a * delta[d]
                coordinates.append(site)
                #FIXME: index_tuple gets added for both sites, hence not unique labelling sites correctly -> should be int 
                # or label via connectivity matrix
                labels.append((self.sublattice_labels[s],) + index_tuple) 
        self.coordinates, self.labels  = coordinates, labels
        print
        self._set_connectivity(lattice_constant=a, nn_factor=parser.lattice_constant.nn_factor)
    
    def _set_connectivity(self, lattice_constant, nn_factor, tol=1e-12):
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
        coordinates = self.coordinates
        N = len(coordinates)     
        C = np.zeros((N, N), dtype=int)
        for i in range(N):
            for j in range(i + 1, N):
                # Sum of squared distances
                dist_sq = 0.0
                for d in range(self.n_dim):
                    diff = coordinates[i][d] - coordinates[j][d]
                    dist_sq += diff * diff
                dist = np.sqrt(dist_sq)
                # Nearest Neighbours
                if abs(dist - (nn_factor * lattice_constant)) < tol:
                    C[i, j] = 1
                    C[j, i] = 1
        self.connectivity_matrix = C

    def plot_lattice(self, list_indexes:bool=False):
        """
        Plots a 2D or 3D projection of an n-dimensional lattice structure,
        coloring each sublattice differently.
        """
        coordinates = self.coordinates
        C = getattr(self, "connectivity_matrix", None)
        labels = getattr(self, "labels", None)

        if not coordinates:
            print("No coordinates to plot.")
            return

        n_dim = self.n_dim
        if n_dim not in (2, 3):
            raise ValueError("This example only handles 2D or 3D plots.")

        # Convert each coordinate to the list of all dims (for 2D or 3D plotting)
        coords_proj = [list(coord) for coord in coordinates]
        fig = plt.figure(figsize=(6, 6))
        if n_dim == 2:
            ax = fig.add_subplot(111)
        else:  # n_dim == 3
            ax = fig.add_subplot(111, projection='3d')

        # ------------------------------------------
        # 1) Identify sublattices and assign colors
        # ------------------------------------------
        if labels is not None:
            sublattices = [lab[0] for lab in labels if isinstance(lab, tuple)]
            unique_sublattices = sorted(set(sublattices))
        else:
            unique_sublattices = ["All"]

        color_cycle = cycle(["blue", "yellow", "red", "orange", "purple", "cyan"])
        sublattice_colors = {sub: next(color_cycle) for sub in unique_sublattices}
        # ------------------------------------------
        # 2) Plot each sublattice separately
        # ------------------------------------------
        for sub in unique_sublattices:
            sub_indices = []
            if labels is not None:
                sub_indices = [i for i, lab in enumerate(labels) 
                            if isinstance(lab, tuple) and lab[0] == sub]
            else:
                sub_indices = range(len(coordinates))
            sub_coords = [coords_proj[i] for i in sub_indices]
            if not sub_coords:
                continue

            xvals = [p[0] for p in sub_coords]
            yvals = [p[1] for p in sub_coords]
            if n_dim == 2:
                ax.scatter(
                    xvals, yvals,
                    c=sublattice_colors[sub],
                    s=20, alpha=0.9,
                    label=sub
                )
            else:
                zvals = [p[2] for p in sub_coords]
                ax.scatter(
                    xvals, yvals, zvals,
                    c=sublattice_colors[sub],
                    s=20, alpha=0.9,
                    label=sub
                )
        # ------------------------------------------
        # 3) Plot connectivity lines if available
        # ------------------------------------------
        if C is not None:
            N = len(coordinates)
            for i in range(N):
                for j in range(i + 1, N):
                    if C[i, j] == 1:
                        xi, yi = coords_proj[i][0], coords_proj[i][1]
                        xj, yj = coords_proj[j][0], coords_proj[j][1]

                        if n_dim == 2:
                            ax.plot([xi, xj], [yi, yj], 'k-', lw=1, alpha=0.5)
                        else:
                            zi = coords_proj[i][2]
                            zj = coords_proj[j][2]
                            ax.plot([xi, xj], [yi, yj], [zi, zj], 'k-', lw=1, alpha=0.5)
        # ------------------------------------------
        # 4) (Optional) Add text labels
        # ------------------------------------------
        if list_indexes:
            for idx, coord_nd in enumerate(coordinates):
                # e.g. label might be ("A", 2, 3)
                label_tuple = labels[idx]
                if isinstance(label_tuple, tuple) and len(label_tuple) >= 2:
                    # sublattice = label_tuple[0]
                    index_tuple = label_tuple[1:]
                    label_str = f"{index_tuple}" #f"{sublattice}{index_tuple}"
                else:
                    label_str = str(label_tuple)

                if n_dim == 2:
                    ax.text(coord_nd[0], coord_nd[1], label_str, color='red',
                            fontsize=8, ha='center', va='center')
                else:
                    ax.text(coord_nd[0], coord_nd[1], coord_nd[2], label_str,
                            color='red', fontsize=8, ha='center', va='center')
        if n_dim == 2:
            ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"{self.name} Lattice")
        ax.legend(loc='upper right')
        plt.show()

    