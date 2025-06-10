import numpy as np

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

    def get_chern_invariant(self, band: int = 0, tol=1e-6):
        """
        Compute the Chern invariant for a single band using
        the Fukui-Hatsugai-Suzuki discretization on a N_x N_y k-grid.
        """
        assert(self.model_options.location in ["both", "bulk"])
        geometry = self.geometry
        N_k = geometry.N_k
        k_x = geometry.kx_bulk
        k_y = geometry.ky_bulk
        U_k_dict = self.tight_binding.U_k_dict
        # Chern Invariant
        print(f"Calculating Chern Invariant...")
        F = np.zeros((N_k-1, N_k-1)) # Berry Flux
        for i in range(N_k - 1):
            for j in range(N_k - 1):
                k = (k_x[i],    k_y[j]) # current
                k_x_p = (k_x[i+1], k_y[j]) # right
                k_y_p = (k_x[i],   k_y[j+1]) # up
                k_xy_pp = (k_x[i+1], k_y[j+1]) # diagonal-up-right
                # Band Eigenstates
                u = U_k_dict[f"[{k[0]},{k[1]}]"][:, band]
                u_x = U_k_dict[f"[{k_x_p[0]},{k_x_p[1]}]"][:, band]
                u_y = U_k_dict[f"[{k_y_p[0]},{k_y_p[1]}]"][:, band]
                u_xy= U_k_dict[f"[{k_xy_pp[0]},{k_xy_pp[1]}]"][:, band]
                # Link U_n(k) = <u(k)|u(k+dn)>
                S_1 = np.dot(u, u_x)
                S_2 = np.dot(u_x, u_xy)
                S_3 = np.dot(u_xy, u_y)
                S_4 = np.dot(u_y, u)
                U_1 = self._phase(S_1, tol)
                U_2 = self._phase(S_2, tol)
                U_3 = self._phase(S_3, tol)
                U_4 = self._phase(S_4, tol)
                # Berry flux F = Arg( U1 * U2 * U3 * U4 )
                F_ij = np.angle(U_1 * U_2 * U_3 * U_4)
                F[i, j] = F_ij
        C = F.sum() / (2 * np.pi)
        print(f"Chern Invariant - Done!")
        return C, F
    
    def _phase(self, S, tol):
        norm = np.abs(S)
        return S / norm if norm > tol else 1.0
