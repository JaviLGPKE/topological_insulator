import numpy as np
from matplotlib import pyplot as plt

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
        the Fukui-Hatsugai-Suzuki discretization on a N_x, N_y k-grid.
        """
        assert self.model_options.location in ["both", "bulk"]
        geom = self.geometry
        N_k = geom.N_k
        kx = geom.kx_bulk
        ky = geom.ky_bulk
        U_k = self.tight_binding.U_k_dict

        F = np.zeros((N_k, N_k), dtype=float)
        for i in range(N_k):
            ip = (i + 1) % N_k # periodic BC k_x wrap-around
            for j in range(N_k):
                jp = (j + 1) % N_k # periodic BC k_y wrap-around
                u = U_k[f"[{kx[i]},{ky[j]}]"][:, band]
                u_x = U_k[f"[{kx[ip]},{ky[j]}]"][:, band]
                u_y = U_k[f"[{kx[i]},{ky[jp]}]"][:, band]
                u_xy = U_k[f"[{kx[ip]},{ky[jp]}]"][:, band]
                # Links: U_n(1) = ⟨u|u_+dn⟩/|⟨u|u+dn⟩|
                U1 = self._phase(np.vdot(u,   u_x), tol)
                U2 = self._phase(np.vdot(u_x, u_xy), tol)
                U3 = self._phase(np.vdot(u_xy,u_y), tol)
                U4 = self._phase(np.vdot(u_y,   u), tol)
                # Berry flux: F = Arg( U1 * U2 * U3 * U4 )
                F[i, j] = np.angle(U1 * U2 * U3 * U4)
        C = F.sum() / (2 * np.pi)
        return np.round(C), F

    
    def _phase(self, S, tol):
        norm = np.abs(S)
        return (S / norm) if norm > tol else (1 + 0j)
    
    def plot_berry_flux(self, F:np.ndarray=None):
        g = self.geometry
        k_x, k_y = g.kx_bulk, g.ky_bulk
        KX_full, KY_full = np.meshgrid(k_x, k_y, indexing='ij')
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(
            KX_full, KY_full, F, 
            rcount=F.shape[0], ccount=F.shape[1],
            linewidth=0, antialiased=True
        )
        ax.set_xlabel(r'$k_x$')
        ax.set_ylabel(r'$k_y$')
        ax.set_zlabel(r'$F$')
        plt.tight_layout()
        plt.show()