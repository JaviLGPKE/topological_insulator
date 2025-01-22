import math
import numpy as np
import matplotlib.pyplot as plt

def honeycomb_coords(Nx, Ny, a0=1.0):
    coords, labels = [], []
    a1 = (1.5,  math.sqrt(3)/2)
    a2 = (1.5, -math.sqrt(3)/2)
    deltaA = (0.0, 0.0)
    deltaB = (0.5, math.sqrt(3)/2)
    for i in range(Nx):
        for j in range(Ny):
            xA = a0*(i*a1[0] + j*a2[0] + deltaA[0])
            yA = a0*(i*a1[1] + j*a2[1] + deltaA[1])
            coords.append((xA, yA))
            labels.append(("A", i, j))  # label for sublattice A
            xB = a0*(i*a1[0] + j*a2[0] + deltaB[0])
            yB = a0*(i*a1[1] + j*a2[1] + deltaB[1])
            coords.append((xB, yB))
            labels.append(("B", i, j))  # label for sublattice B
    return coords, labels

def build_adjacency(coords, a0=1.0, tol=1e-7):
    N = len(coords)
    C = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if abs(dist - a0) < tol:   # nearest neighbors
                C[i, j] = 1
                C[j, i] = 1
    return C

def plot_lattice(coords, A, labels=None):
    xvals = [c[0] for c in coords]
    yvals = [c[1] for c in coords]
    plt.figure(figsize=(6,6))
    plt.scatter(xvals, yvals, c='blue', s=20, alpha=0.9, zorder=2)
    N = len(coords)
    for i in range(N):
        for j in range(i+1, N):
            if A[i, j] == 1:
                xi, yi = coords[i]
                xj, yj = coords[j]
                plt.plot([xi, xj], [yi, yj], 'k-', lw=1, alpha=0.5, zorder=1)
    if labels is not None:
        for idx, (x, y) in enumerate(coords):
            (sub, i, j) = labels[idx]   # e.g. ("A", 0, 0)
            plt.text(x, y, f"{sub}({i},{j})", color='red', fontsize=8,
                     horizontalalignment='center', verticalalignment='center')
    plt.axis("equal")
    plt.title("Honeycomb Lattice")
    plt.show()


all_sites, labels = honeycomb_coords(Nx=10, Ny=10, a0=1.0)
C = build_adjacency(all_sites, a0=1.0, tol=1e-7)
plot_lattice(all_sites, C, labels)
