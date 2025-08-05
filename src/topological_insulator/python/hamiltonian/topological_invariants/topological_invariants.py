import numpy as np
from matplotlib import pyplot as plt
from pfapack import pfaffian as pf

from ..notation import Notation
from ...model_options import ModelOptions
from ...cell_parser import CellParser
from ...geometry import Geometry
from ..tight_binding.bulk_tb import TightBinding

from IPython import embed

class TopologicalInvariants(Notation):
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
        zak_phase = 0
        for i in range(1, geometry.N_k):
            k, k_0 = geometry.k_edge[i], geometry.k_edge[i-1]
            u_k = U_k_dict[f"{k}"][:, band]
            u_k_0 = U_k_dict[f"{k_0}"][:, band]
            S = np.vdot(u_k_0, u_k)
            zak_phase += 1j * np.log(S/np.abs(S)) # phase = ln(e^(i*phase))
        print(f"Zak Phase - Done!")
        return zak_phase

    def get_topological_invariant(self, bands, tol=1e-6):
        assert self.model_options.location in ["both", "bulk"]
        if not np.isclose(self.cell_parser.field.magnetic.value, 0, rtol=tol):
            return self.abelian_chern_invariant(bands, tol)
        else:
            return self.Z2_invariant(bands)

    def Z2_invariant(self, bands=[]):
        print(f"Calculating Z2 Invariant...")
        g, tb = self.geometry, self.tight_binding
        if bands == []:
            N_bands = len(tb.sublattice_idxs) * len(tb.coupled_states)
            bands = [i for i in range(N_bands//2)]
        O = tb.O # Time-Reversal Operator
        U_k = tb.U_k_dict
        kx, ky = g.kx_bulk, g.ky_bulk
        trims = g.trims
        deltas = []
        for k in trims:
            i = np.argmin(np.abs(g.kx_bulk - k[0]))
            j = np.argmin(np.abs(g.ky_bulk - k[1]))
            key = f"[{kx[i]}, {ky[j]}]"
            u_k = U_k[key][:, bands]
            w_k = u_k.conj().T @ O @ u_k.conj()
            w_k_det = np.linalg.det(w_k)
            P_k = pf.pfaffian(w_k)
            delta_i = np.sqrt(w_k_det) / P_k
            deltas.append(np.sign(delta_i.real))
        total_product = np.prod(deltas)
        Z_2 = int((1 - total_product) / 2) # maps +1 to 0, −1 to 1
        print(f"Z2 Invariant - Done!")
        return Z_2

    def abelian_chern_invariant(self, bands, tol):
        band = 0 if bands == [] else bands[0]
        print(f"Calculating Chern Invariant...")
        geometry = self.geometry
        N_k = geometry.N_k
        kx = geometry.kx_bulk
        ky = geometry.ky_bulk
        U_k = self.tight_binding.U_k_dict
        # Berry Curvature
        F, F_dict = np.zeros((N_k, N_k), dtype=float), {}
        for i in range(N_k):
            ip = (i + 1) % N_k # periodic BC
            for j in range(N_k):
                jp = (j + 1) % N_k # periodic BC
                u = U_k[f"[{kx[i]}, {ky[j]}]"][:, band]
                u_x = U_k[f"[{kx[ip]}, {ky[j]}]"][:, band]
                u_y = U_k[f"[{kx[i]}, {ky[jp]}]"][:, band]
                u_xy = U_k[f"[{kx[ip]}, {ky[jp]}]"][:, band]
                U_1 = self._phase(np.vdot(u, u_x))
                U_2 = self._phase(np.vdot(u_x, u_xy))
                U_3 = self._phase(np.vdot(u_xy, u_y))
                U_4 = self._phase(np.vdot(u_y, u))
                # F_dict[f"{i}, {j}"] = [U_1, U_2, U_3, U_4] # NOTE: Debugging
                F[i, j] = np.angle(U_1 * U_2 * U_3 * U_4)
        C = F.sum() / (2 * np.pi)
        print(f"Chern Invariant - Done!")
        return C, F#, F_dict
        
    def _phase(self, S):
        norm = np.abs(S)
        return (S / norm)

    def get_local_density_of_states(self, site_idx:int = 0, band:int=None,
                                 E_max=10, E_min=-10,  N_E=1000, eta:float=1e-1,):
        geometry = self.geometry
        tb = self.tight_binding
        k_edge = geometry.k_edge
        if band == None:
            band = tb.get_edge_bands(geometry)
        N_projections = len(tb.coupled_states)
        N_sites = len(tb.sublattice_idxs)
        N_bands = N_projections * N_sites
        E_vals = np.linspace(E_min, E_max, N_E)
        Psi_dict = tb.band_structure_data["eigenvector_dict"]
        LDOS = np.zeros_like(E_vals)
        for k_idx, k in enumerate(k_edge):
            E_k = tb.E_k_dict[f"{k}"] 
            for n in range(N_bands):
                Psi_k = Psi_dict[n][k_idx, :]
                start = site_idx * N_projections
                end   = (site_idx + 1)*N_projections
                c_i = Psi_k[start:end]
                weight = np.sum(np.abs(c_i)**2)
                LDOS += weight * self._lorentz(E_vals, E_k[n], eta)
        LDOS /= len(k_edge)
        return LDOS, E_vals
    
    def _lorentz(self, E, E0, eta):
        return (1/np.pi) * (eta / ((E - E0)**2 + eta**2))
    
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
    
    def plot_density_of_states(self, LDOS:np.ndarray, E_vals:np.ndarray):
        plt.plot(E_vals, LDOS)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Density of States (a.u)")
        plt.show()