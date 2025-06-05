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
    
    # def build_wavefunction(self):
    #     for uncoupled_state, CG in self.CG_coefficients.items():
    #         l   = self.get_quantum_number(uncoupled_state, pos=0)
    #         m_l = self.get_quantum_number(uncoupled_state, pos=1)
    #         m_s = self.get_quantum_number(uncoupled_state, pos=3)
    #         j = l + 1/2
    #         m_j = m_l + m_s
    #         orbitals = self.l_to_orbitals(l, m_l)
    #         coupled_state = f"|{j},{m_j}>"
    #         self.coupled_states[coupled_state] = []
    #         for orbital, coefficient in orbitals.items():
    #             K = CG * coefficient
    #             self.coupled_states[coupled_state].append(
    #                     lambda phi, theta, K=K, : K * Y_m_l(m_l, l, phi, theta)
    #             )
    
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
        zak_phase = 0
        dim_slice = slice(idx * N_projections, (idx + 1) * N_projections)
        for i in range(1, geometry.N_k): 
            k, k_0 = geometry.k_edge[i], geometry.k_edge[i-1]
            band = 1
            u_k = U_k_dict[f"{k}"][:, band]
            u_k_0 = U_k_dict[f"{k_0}"][:, band]
            S = np.vdot(u_k_0, u_k)
            zak_phase += 1j * np.log(S/np.abs(S)) # phase = log(e^(i*phase))
        v = zak_phase / (2*np.pi)
        return v.real