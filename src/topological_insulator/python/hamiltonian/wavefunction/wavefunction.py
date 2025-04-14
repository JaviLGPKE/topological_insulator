import numpy as np
from scipy.special import sph_harm as Y_m_l
from scipy import integrate
from typing import Union

from ..notation import Notation
from ...model_options import ModelOptions
from ...cell_parser import CellParser
from ...geometry import Geometry
from ..tight_binding.bulk_tb import TightBinding

from IPython import embed

class WaveFunction(Notation):
    def __init__(self, model_options:ModelOptions, cell_parser:CellParser, tight_binding:TightBinding):
        super().__init__()
        # Arguments
        self.model_options = model_options
        self.cell_parser = cell_parser
        self.tight_binding = tight_binding
        # Eigenstates
        self.coupled_states = {}
    
    def build_wavefunction(self):
        for uncoupled_state, CG in self.CG_coefficients.items():
            l   = self.get_quantum_number(uncoupled_state, pos=0)
            m_l = self.get_quantum_number(uncoupled_state, pos=1)
            m_s = self.get_quantum_number(uncoupled_state, pos=3)
            j = l + 1/2
            m_j = m_l + m_s
            orbitals = self.l_to_orbitals(l, m_l)
            coupled_state = f"|{j},{m_j}>"
            self.coupled_states[coupled_state] = []
            for orbital, coefficient in orbitals.items():
                K = CG * coefficient
                self.coupled_states[coupled_state].append(
                        lambda phi, theta, K=K, : K * Y_m_l(m_l, l, phi, theta)
                )
    
    def evaluate_wavefunction(self, coupled_state, phi, theta):
        return sum(f(phi, theta) for f in self.coupled_states[coupled_state])

    def calculate_chern_invariant(self, geometry:Geometry, U_k_dict:dict, N_0=20, N_p=20):
        if self.model_options.solve_connectivity:
            return None
        print(f"Calculating Chern Invariant...")
        sublattice_idxs = self.tight_binding.sublattice_idxs
        N_projections = 6
        idx_map = {idx: pos for pos, idx in enumerate(sublattice_idxs)}
        idx = idx_map[self.tight_binding.edge_idxs[0]]
        # Chern Invariant
        v = 0
        dim_slice = slice(idx * N_projections, (idx + 1) * N_projections)
        for i in range(1, geometry.N_k): 
            k, k_0 = geometry.k_edge[i], geometry.k_edge[i-1]
            U_n_k = U_k_dict[f"{k}"][:, 0]
            U_n_k_0 = U_k_dict[f"{k_0}"][:, 0]
            U_k = U_n_k[dim_slice]
            U_k_0 = U_n_k_0[dim_slice]
            dU_k = U_k - U_k_0
            # Line Integral -> dk ~ k*(d0^2 + sin^2(0)dp^2)^1/2
            print(i, geometry.N_k)
            for d0 in np.linspace(0, np.pi, N_0):
                for dp in np.linspace(0, 2*np.pi, N_p):
                    basis = np.array([self.evaluate_wavefunction(state, d0, dp) for state in self.coupled_states.keys()])
                    dk = k*(d0**2 + (np.sin(d0) * dp)**2)**1/2
                    Psi_k = U_k * basis
                    A_k = 0
                    if basis.all() != 0:
                        A_k = 1j* Psi_k.conj().T @ (dU_k @ basis)
                    v += A_k * dk
        v /= 2*np.pi
        return v

    # def _berry_connection(self, U_k, dU_k, d0, dp):
    #     A_k = 0
    #     for n, uncoupled_bra in enumerate(self.CG_coefficients.keys()):
    #         l_n   = self.get_quantum_number(uncoupled_bra, pos=0)
    #         m_l_n = self.get_quantum_number(uncoupled_bra, pos=1)
    #         m_s_n = self.get_quantum_number(uncoupled_bra, pos=3)
    #         j_n = l_n + 1/2
    #         m_j_n = m_l_n + m_s_n
    #         bra_state = f"|{j_n},{m_j_n}>"
    #         bra_coeff = self.evaluate_wavefunction(bra_state, d0, dp)
    #         for m, uncoupled_ket in enumerate(self.CG_coefficients.keys()):
    #             l_m   = self.get_quantum_number(uncoupled_ket, pos=0)
    #             m_l_m = self.get_quantum_number(uncoupled_ket, pos=1)
    #             m_s_m = self.get_quantum_number(uncoupled_ket, pos=3)
    #             j_m = l_m + 1/2
    #             m_j_m = m_l_m + m_s_m
    #             ket_state = f"|{j_m},{m_j_m}>"
    #             ket_coeff = self.evaluate_wavefunction(ket_state, d0, dp)
    #             U_k_nm, dU_k_nm = U_k[n][m], dU_k[n][m]
    #             if U_k_nm == 0 or dU_k_nm == 0:
    #                 continue
    #             A_k += U_k_nm * bra_coeff * dU_k_nm * ket_coeff
    #     # print("berry_connection")
    #     # embed()
    #     return A_k