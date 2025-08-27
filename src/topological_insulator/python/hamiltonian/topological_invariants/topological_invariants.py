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
        assert(self.tight_binding.location == "edge")
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

    def Z2_invariant(self, bands=[], print_deltas:bool=False):
        assert(self.tight_binding.location == "bulk")
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
            if print_deltas:
                print(f"k={k}: delta = {np.sign(delta_i.real)}")
        total_product = np.prod(deltas)
        Z_2 = int((1 - total_product) / 2) # maps +1 to 0, −1 to 1
        print(f"Z2 Invariant - Done!")
        return Z_2

    def abelian_chern_invariant(self, bands):
        assert(self.tight_binding.location == "bulk")
        band = 0 if bands == [] else bands[0]
        print(f"Calculating Chern Invariant...")
        geometry = self.geometry
        N_k = geometry.N_k
        kx = geometry.kx_bulk
        ky = geometry.ky_bulk
        U_k = self.tight_binding.U_k_dict
        # Berry Curvature
        F = np.zeros((N_k, N_k), dtype=float)
        for i in range(N_k):
            ip = (i + 1) % N_k # periodic BC
            for j in range(N_k):
                jp = (j + 1) % N_k # periodic BC
                u = U_k[f"[{kx[i]}, {ky[j]}]"][:, band]
                u_x = U_k[f"[{kx[ip]}, {ky[j]}]"][:, band]
                u_y = U_k[f"[{kx[i]}, {ky[jp]}]"][:, band]
                u_xy = U_k[f"[{kx[ip]}, {ky[jp]}]"][:, band]
                U_1 = self._abelian_phase(np.vdot(u, u_x))
                U_2 = self._abelian_phase(np.vdot(u_x, u_xy))
                U_3 = self._abelian_phase(np.vdot(u_xy, u_y))
                U_4 = self._abelian_phase(np.vdot(u_y, u))
                F[i, j] = np.angle(U_1 * U_2 * U_3 * U_4)
        C = F.sum() / (2 * np.pi)
        print(f"Chern Invariant - Done!")
        return C, F
    
    def _abelian_phase(self, S):
        norm = np.abs(S)
        return (S / norm)
    
    def non_abelian_chern_invariant(self, bands):
        """
        Fukui-Hatsugai non-Abelian Chern number for a set of occupied bands.
        Returns (C, F) where F[i,j] is the flux (radians) on each plaquette and
        C is the total Chern number (sum(F)/(2*pi)).
        """
        assert(self.tight_binding.location == "bulk")
        print("Calculating non-Abelian Chern Invariant...")
        geometry = self.geometry
        N_k = geometry.N_k
        kx = geometry.kx_bulk
        ky = geometry.ky_bulk
        U_k = self.tight_binding.U_k_dict
        sample_key = f"[{kx[0]}, {ky[0]}]"
        N_bands = U_k[sample_key].shape[1]
        N_occ_bands = list(range(N_bands)) if bands == [] else list(bands)
        F = np.zeros((N_k, N_k), dtype=float)
        for i in range(N_k):
            ip = (i + 1) % N_k
            for j in range(N_k):
                if not geometry.BZ_mask[i, j]:
                    continue
                jp = (j + 1) % N_k
                U00 = U_k[f"[{kx[i]}, {ky[j]}]"][:, N_occ_bands]
                U10 = U_k[f"[{kx[ip]}, {ky[j]}]"][:, N_occ_bands]
                U11 = U_k[f"[{kx[ip]}, {ky[jp]}]"][:, N_occ_bands]
                U01 = U_k[f"[{kx[i]}, {ky[jp]}]"][:, N_occ_bands]
                # M_x(k)_{mn} = <u_m(k) | u_n(k+dx)>
                M1 = np.conj(U00).T @ U10
                M2 = np.conj(U10).T @ U11
                M3 = np.conj(U11).T @ U01
                M4 = np.conj(U01).T @ U00
                U1 = self._non_abelian_phase(M1)
                U2 = self._non_abelian_phase(M2)
                U3 = self._non_abelian_phase(M3)
                U4 = self._non_abelian_phase(M4)
                F[i, j] = np.angle( U1 * U2 * U3 * U4)
        C = F.sum() / (2 * np.pi)
        print("Non-Abelian Chern Invariant - Done!")
        return C, F
    
    def _non_abelian_phase(self, M):
        det = np.linalg.det(M)
        return det / np.abs(det)

    def get_density_of_states(self,
                                 E_max=10, E_min=-10,  N_E=1000, eta:float=1e-1):
        assert(self.tight_binding.location == "edge")
        geometry = self.geometry
        tb = self.tight_binding
        k_edge = geometry.k_edge
        N_projections = len(tb.coupled_states)
        N_bands = len(tb.sublattice_idxs) * N_projections
        E = np.linspace(E_min, E_max, N_E)
        DOS = np.zeros_like(E)
        for i in range(len(self.sublattice_idxs)):
            for k_idx, k in enumerate(k_edge):
                E_k = tb.E_k_dict[f"{k}"] 
                for n in range(N_bands):
                    weight = tb.weight(k_idx, i, n)
                    DOS += weight * self._lorentz(E, E_k[n], eta)
        DOS /= len(k_edge)
        return E, DOS
    
    def get_local_density_of_states(self, site_idx:int = 0,
                                 E_max=10, E_min=-10,  N_E=1000, eta:float=1e-1):
        assert(self.tight_binding.location == "edge")
        geometry = self.geometry
        tb = self.tight_binding
        k_edge = geometry.k_edge
        N_projections = len(tb.coupled_states)
        N_bands = len(tb.sublattice_idxs) * N_projections
        E = np.linspace(E_min, E_max, N_E)
        LDOS = np.zeros_like(E)
        for k_idx, k in enumerate(k_edge):
            E_k = tb.E_k_dict[f"{k}"] 
            for n in range(N_bands):
                weight = tb.weight(k_idx, site_idx, n)
                LDOS += weight * self._lorentz(E, E_k[n], eta)
        LDOS /= len(k_edge)
        return E, LDOS
    
    def _lorentz(self, E, E0, eta):
        return (1/np.pi) * (eta / ((E - E0)**2 + eta**2))

    def get_band_gap(self, n, m, only_dE:bool=True, ):
        assert(self.tight_binding.location == "bulk")
        geometry = self.geometry
        tb = self.tight_binding
        # NOTE: numpy.linalg.eigh returns eigenvalues in ascending order
        E_0 = tb.E_k_dict[f"{geometry.K_point}"][n]
        E_1 = tb.E_k_dict[f"{geometry.Gamma}"][m]
        dE = E_0 - E_1
        if only_dE:
            return dE
        else:
            return dE, E_0, E_1
    
    def plot_berry_flux(self, F:np.ndarray=None):
        geometry = self.geometry
        k_x, k_y = geometry.kx_bulk, geometry.ky_bulk
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

    def plot_density_of_states(self, E_vals: np.ndarray,
                            DOS: np.ndarray,
                            figsize: tuple = (8, 7),
                            annotate_max: bool = True,
                            xlabel:str = "LDOS"):
        """
        Plot LDOS/DOS vs. energy and, optionally, mark the energy at which LDOS is maximal,
        all on the same axes.
        """
        idx_max = np.argmax(DOS)
        E_peak = E_vals[idx_max]
        fig, ax = plt.subplots(figsize=figsize)
        # ax.plot(E_vals, LDOS, color="k", lw=1.8)
        ax.plot(DOS, E_vals, color="k", lw=1.8)
        if annotate_max:
            ax.axhline(
                E_peak,
                color="r",
                linestyle='--',
                linewidth=1.5,
                label=f'Max {xlabel} at {E_peak:.2f} eV'
            )
        ax.set_ylabel("Energy (eV)", fontsize=12)
        ax.set_xlabel(f"{xlabel} (a.u.)", fontsize=12)
        if xlabel == "LDOS":
            ax.set_title("Local Density of States", fontsize=14)
        else:
            ax.set_title("Total Density of States", fontsize=14)
        ax.legend(frameon=True)
        ax.grid(True, ls=':', alpha=0.6)

        plt.tight_layout()
        plt.show()
