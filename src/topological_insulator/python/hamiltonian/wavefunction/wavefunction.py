import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from pfapack import pfaffian as pf

from ..notation import Notation
from ...model_options import ModelOptions
from ...cell_parser import CellParser
from ...geometry import Geometry
from ..tight_binding.bulk_tb import TightBinding

from IPython import embed

class WaveFunction(Notation):
    def __init__(self, model_options:ModelOptions, cell_parser:CellParser, 
                 geometry:Geometry, tight_binding:TightBinding):
        super().__init__()
        # Arguments
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.geometry = geometry
        self.tight_binding = tight_binding
        # Parity

    def get_zak_phase(self, band = 1):
        assert(self.model_options.location in ["both", "edge"])
        geometry = self.geometry
        U_k_dict = self.tight_binding.U_k_dict
        print(f"Calculating Zak Phase...")
        # Chern Invariant
        zak_phase = 0
        for i in range(1, geometry.N_k):
            k, k_0 = geometry.k_edge[i], geometry.k_edge[i-1]
            u_k = U_k_dict[f"{k}"][:, band]
            u_k_0 = U_k_dict[f"{k_0}"][:, band]
            S = np.vdot(u_k_0, u_k)
            zak_phase += 1j * np.log(S/np.abs(S)) # phase = log(e^(i*phase))
        print(f"Zak Phase - Done!")
        return zak_phase

    def get_topological_invariant(self, band=None, tol=1e-6):
        assert self.model_options.location in ["both", "bulk"]
        if not np.isclose(self.cell_parser.field.magnetic.value, 0, rtol=tol):
            return self.abelian_chern_invariant(band, tol)
        else:
            return self.Z2_invariant()

    def Z2_invariant(self, E_f = 0, filling_factor=1):
        # NOTE: TR invariant is ill defined for gapless dispersions.
        g, tb = self.geometry, self.tight_binding
        O = tb.O # Time-Reversal Operator
        U_k = tb.U_k_dict
        kx, ky = g.kx_bulk, g.ky_bulk
        trims = g.trims
        deltas = []
        for k in trims:
            i = np.argmin(np.abs(g.kx_bulk - k[0]))
            j = np.argmin(np.abs(g.ky_bulk - k[1]))
            key = f"[{kx[i]},{ky[j]}]"
            u_k = U_k[key]
            w_k = u_k.conj().T @ O @ u_k.conj()
            w_k_det = np.linalg.det(w_k)
            P_k = pf.pfaffian(w_k)
            delta_i = np.sqrt(w_k_det) / P_k
            deltas.append(np.sign(delta_i.real))
        total_product = np.prod(deltas)
        Z_2 = int((1 - total_product) / 2)  # maps +1 to 0, −1 to 1
        return Z_2

    def abelian_chern_invariant(self, band, tol):
        geom = self.geometry
        N_k = geom.N_k
        kx = geom.kx_bulk
        ky = geom.ky_bulk
        U_k = self.tight_binding.U_k_dict
        # Berry Curvature
        if band == None:
            band = 0
        F, F_dict = np.zeros((N_k, N_k), dtype=float), {}
        for i in range(N_k):
            ip = (i + 1) % N_k # periodic BC
            for j in range(N_k):
                jp = (j + 1) % N_k # periodic BC
                u = U_k[f"[{kx[i]},{ky[j]}]"][:, band]
                u_x = U_k[f"[{kx[ip]},{ky[j]}]"][:, band]
                u_y = U_k[f"[{kx[i]},{ky[jp]}]"][:, band]
                u_xy = U_k[f"[{kx[ip]},{ky[jp]}]"][:, band]
                U_1 = self._phase(np.vdot(u, u_x), tol)
                U_2 = self._phase(np.vdot(u_x, u_xy), tol)
                U_3 = self._phase(np.vdot(u_xy, u_y), tol)
                U_4 = self._phase(np.vdot(u_y, u), tol)
                F_dict[f"{i}, {j}"] = [U_1, U_2, U_3, U_4]
                F[i, j] = np.angle(U_1 * U_2 * U_3 * U_4)
        C = F.sum() / (2 * np.pi)
        return C, F#, F_dict
        
    def _phase(self, S, tol):
        norm = np.abs(S)
        return (S / norm) if norm > tol else (1 + 0j)
    
    def plot_berry_flux(self, F:np.ndarray=None):
        g = self.geometry
        k_x, k_y = g.kx_bulk, g.ky_bulk
        KX_full, KY_full = np.meshgrid(k_x, k_y, indexing='ij')
        fig = plt.figure(figsize=(7,7))
        ax  = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            KX_full, KY_full, F, 
            rcount=F.shape[0], ccount=F.shape[1],
            linewidth=0, antialiased=True
        )
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_zlabel(r'$F$')
        plt.show()