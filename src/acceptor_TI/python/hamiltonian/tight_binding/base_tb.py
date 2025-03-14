import numpy as np
from scipy import linalg
from sympy.physics.quantum.cg import CG
import re
from matplotlib import pyplot as plt
from abc import abstractmethod

from ...model_options import ModelOptions
from ...cell_parser import CellParser
from ...geometry import Geometry

class TightBinding:
    """
    Tight-Binding approximation Hamiltonian that can include, nearest neighbour hopping, 
    spin-orbit coupling interaction and Coulomb repulsive interaction terms.
    """

    def __init__(self, model_options:ModelOptions, cell_parser: CellParser):
        # Arguments
        self.model_options = model_options
        self.cell_parser = cell_parser
        # Orbitals
        self.orbitals = ['s', 'p_x', 'p_y', 'p_z']
        self.direction_index = {'x': 0, 'y': 1, 'z': 2}
        self.state_pattern = re.compile(r'\|([\d\.\-]+),([\d\.\-]+);([\d\.\-]+),([\d\.\-]+)\>')
        self.n_spins = 2
        self.n_orbitals = 4
        self._clebsch_gordan()
        # Sublattice
        self.sublattice_data_dict = {}

    @abstractmethod
    def build_hamiltonian(self, geometry:Geometry) -> None:
        """
        Must build the necessary sublattice data for the 'calculate_eigenvalues' method.
        """
        self.sublattice_data_dict = None
        self.sublattice_connectivity = None
        self.H = None
        raise NotImplementedError("'build_hamiltonian' method not implemented!")

    def sublattice_data(self, geometry:Geometry, location:str, idx:int):
        neighbour_idxs = geometry.get_neighbour_idxs(idx)
        dr_list, dm_list = geometry.get_dr("bulk", idx, neighbour_idxs, type="list")
        dr_dict, dm_dict = geometry.get_dr(location, idx, neighbour_idxs, type="dict")
        directional_cosines = geometry.bond_orientation(dr_list)
        H_ij_dict, coupled_states_dict = self.get_hopping_info(
                    neighbour_idxs, directional_cosines)
        return {
                "idx": idx,
                "neighbour_idxs": neighbour_idxs,
                "dr_dict": dr_dict,
                "dm_dict": dm_dict,
                "hopping_dict": H_ij_dict,
                "coupled_states_dict": coupled_states_dict
            }

    def l_to_orbitals(self, l, m_l):
        """
        Convert an angular momentum state |l, m_l> into a linear combination
        of orbital states. Returns a dictionary where the keys are the orbital
        labels ('s', 'p_x', 'p_y', 'p_z') and the values are the expansion coefficients.
        
        Using the conventions:
        |0,0>         = |s>
        |1,0>         = |p_z>
        |1,+1>        = -1/sqrt(2) ( |p_x> + |p_y> )
        |1,-1>        = +1/sqrt(2) ( |p_x> - |p_y> )
        """
        # s
        if l == 0 and m_l == 0:
            return {"s": 1.0}
        # p
        elif l == 1:
            if m_l == 0:
                return {"p_z": 1.0}
            elif m_l == 1:
                return {"p_x": -1/np.sqrt(2), "p_y": 1j* -1/np.sqrt(2)}
            elif m_l == -1:
                return {"p_x":  1/np.sqrt(2), "p_y": 1j* -1/np.sqrt(2)}
            else:
                raise ValueError("Invalid m_l value for l=1")
        else:
            raise ValueError("Conversion for l > 1 is not implemented!")      

    def get_quantum_number(self, key, pos=0):
        """
        Extract a quantum number from a state string in ket notation: |a,b;c,d>
        The 'pos' parameter determines which quantum number to extract:
            pos = 0: returns the first quantum number (a)
            pos = 1: returns the second quantum number (b)
            pos = 2: returns the third quantum number (c)
            pos = 3: returns the fourth quantum number (d)
        
        Parameters:
            key (str): The state string.
            pos (int): The 0-indexed position of the quantum number to extract.
            
        Returns:
            float: The quantum number at the specified position.
            
        Raises:
            ValueError: If the state string format is incorrect or if 'pos' is out of range.
        """
        match = self.state_pattern.search(key)
        if match:
            try:
                # Adjust for 1-indexed regex groups
                return float(match.group(pos + 1))
            except IndexError:
                raise ValueError(f"State string does not contain a quantum number at position {pos}")
        else:
            raise ValueError("State string format is incorrect.")

    def get_hopping_info(self, neighbour_idxs, directional_cosines):
        H_i = {}
        coupled_states_i = {}
        for neighbour_idx, cosines in zip(neighbour_idxs, directional_cosines):
            j = neighbour_idx
            state_hoppings = self._slater_koster(cosines)
            H_ij, coupled_states = self._get_coupled_hopping(state_hoppings)
            H_i[j] = H_ij
            coupled_states_i[j] = coupled_states
        return H_i, coupled_states_i

    def _clebsch_gordan(self):
        self.CG_coefficients = {}
        j_2 = 1/2
        m_2 = np.arange(-j_2, j_2 + 1, 1)
        for j_1 in [0, 1]:          
            m_1 = np.arange(-j_1, j_1 + 1, 1)
            j_3 = j_1 + j_2
            m_3 = np.arange(-j_3, j_3 + 1, 1)
            for i, m_j in enumerate(m_3):
                cg = 0
                for m_l in m_1:
                    for m_s in m_2:
                        state = f"|{j_1},{m_l};{j_2},{m_s}>"
                        if (m_l + m_s) != m_j:
                            continue
                        self.CG_coefficients[state] = CG(j_1, m_l, j_2, m_s, j_3, m_j).doit()

    def _slater_koster(self, cosines):
        nn_parser = self.cell_parser.eigenvalues.nn_hopping.value
        t_ss = nn_parser["t_ss_sigma"]
        t_sp = nn_parser["t_sp_sigma"]
        t_pp_sigma = nn_parser["t_pp_sigma"]
        t_pp_pi = nn_parser["t_pp_pi"]
        n_z = 0 # TODO: implement bukcling angle in json inbetween sublattices
        l, m = (cosines[0], cosines[1]) if len(cosines) == 2 else (cosines[0], cosines[1])
        n = cosines[2] if len(cosines) == 3 else n_z
        p_cosines = [l, m, n]
        # Case |p_n><p_m| 
        pp_matrix = [
            [
                (p_cosines[i]**2 * t_pp_sigma + (1 - p_cosines[i]**2) * t_pp_pi) if i == j
                else (p_cosines[i] * p_cosines[j] * (t_pp_sigma - t_pp_pi))
                for j in range(3)
            ]
            for i in range(len(p_cosines))
        ]
        state_hoppings = {}
        for alpha in self.orbitals:
            for beta in self.orbitals:
                key = f"|{alpha}><{beta}|"
                # s-s
                if alpha == beta == 's':
                    state_hoppings[key] = t_ss
                # s-p or p-s
                elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                    p_orb = alpha if alpha.startswith('p') else beta
                    d = self.direction_index[p_orb.split('_')[1]]
                    #NOTE: s are symmetric, p orbitals are antisymmetric -> Under spatial inversion:
                    # s(-r) = s(r) and p(-r) = -p(r), hence <s|H|x> = -<x|H|s>
                    t = p_cosines[d] * t_sp
                    if key[1] != "s":
                        t *= -1
                    state_hoppings[key] = t
                # p-p
                elif alpha.startswith('p') and beta.startswith('p'):
                    d1 = self.direction_index[alpha.split('_')[1]]
                    d2 = self.direction_index[beta.split('_')[1]]
                    state_hoppings[key] = pp_matrix[d1][d2]
                else:
                    state_hoppings[key] = 0
        return state_hoppings

    def _get_coupled_hopping(self, state_hoppings:dict):
        """
        Transforms the 8x8 spin-orbit Hamiltonian (s + p orbitals with spin-1/2)
        into the coupled |j, m_j> basis. Returns the transformed 8x8 matrix.
        
        Parameters
        ----------
        state_hoppings : dict
            The Hamiltonian spin-orbit uncoupled state hoppings. 
        
        Returns
        -------
        H_ij : np.ndarray
            The Hamiltonian expressed in the total-angular-momentum coupled basis |j, m_j>.
        """
        dim = len(self.CG_coefficients.keys())
        H_ij = np.zeros(shape=(dim,dim) , dtype=complex)
        # NOTE: Transitions to opposite spin-states are not allowed
        # NOTE: Assumed t_{ss}^{↑} == t_{ss}^{↓}
        coupled_states = {}
        for n, (bra_key, cg_bra) in enumerate(self.CG_coefficients.items()):
            bra_l   = self.get_quantum_number(bra_key, pos=0)
            bra_m_l = self.get_quantum_number(bra_key, pos=1)
            bra_m_s = self.get_quantum_number(bra_key, pos=3)
            bra_orbitals = self.l_to_orbitals(bra_l, bra_m_l)
            j_n = bra_l + 1/2
            m_j_n = bra_m_l + bra_m_s
            for m, (ket_key, cg_ket) in enumerate(self.CG_coefficients.items()):
                ket_l   = self.get_quantum_number(ket_key, pos=0)
                ket_m_l = self.get_quantum_number(ket_key, pos=1)
                ket_m_s = self.get_quantum_number(ket_key, pos=3)
                if bra_m_s != ket_m_s:
                    continue
                j_m = ket_l + 1/2
                m_j_m = ket_m_l + ket_m_s
                ket_orbitals = self.l_to_orbitals(ket_l, ket_m_l)
                t_nm = 0
                for bra_orb, bra_coeff in bra_orbitals.items():
                    for ket_orb, ket_coeff in ket_orbitals.items():
                        hopping_key = f"|{bra_orb}><{ket_orb}|"
                        t_hop = state_hoppings[hopping_key]
                        t_nm += cg_bra * cg_ket * bra_coeff * ket_coeff * t_hop
                        # if j_n == 3/2:
                        #     t_nm+= delta_heavy
                        # elif j_n = 1/2:
                        #     t_nm += delta_light
                # TODO: add spin-orbit coupling hopping "delta"
                coupled_states[f"|{j_n},{m_j_n}><{j_m},{m_j_m}|"] = t_nm
                H_ij[n, m] = t_nm
        # FIXME: sort the matrix into a more predictable state, as currently,
        # i, j = 0, 0 would be interpreted as the |j=1/2, m=-1/2><j=1/2, m=-1/2| entry
        # i, j = 0, 1 would be interpreted as the |j=1/2, m=-1/2><j=3/2, m=-3/2| entry
        # i, j = 1, 0 would be interpreted as the |j=1/2, m=-1/2><j=3/2, m=-1/2| entry
        # i, j = 1, 1 would be interpreted as the |j=1/2, m=+1/2><j=1/2, m=+1/2| entry
        # i, j = 0, 2 would be interpreted as the |j=1/2, m=+1/2><j=3/2, m=-3/2| entry
        # i.e. unpredictable
        # NOTE: maybe include make a unitary transformation matrix
        return H_ij, coupled_states

    @abstractmethod
    def solve_eigenvalues(self, geometry:Geometry, acceptor:bool, H_type:str):
        """
        Must calculate the necessary eigenvalues depending on the requested 
        Hamiltonian type.
        """
        if H_type == "real_space":
            self.E = None
        elif H_type == "reciprocal_space":
            self.E_k_dict = None
        raise NotImplementedError("'solve_eigenvalues' method not implemented")

    def _solve_eigenvalues(self, H):
        E, U = linalg.eigh(H, check_finite=False, driver="evr")
        return E, U

    def _visualise_matrix(self, M):
        plt.figure(figsize=(12, 5))
        cmap = plt.get_cmap('coolwarm')
        cmap.set_bad('white')
        mask = np.isclose(M, 0, atol=1e-12)
        # M_masked = np.ma.masked_where(mask, M)
        plt.subplot(1, 2, 1)
        plt.imshow(M.real, cmap=cmap)
        plt.title('Real Part')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(M.imag, cmap=cmap)
        plt.title('Imaginary Part')
        plt.colorbar()
        plt.show()

    @abstractmethod
    def plot_dispersion(self, geometry: Geometry):
        raise NotImplementedError("Implement dispersion plot method!")

    def plot_band_structure(self, geometry: Geometry):
        # TODO:
        # assert(self.model_options.band_structure)
        print("Plotting band structure...")
        # N_k = geometry.N_k
        # k_vec = geometry.k_path
        # Eplus = self.E_plus_band  
        # Eminus = self.E_minus_band 

        # # Create a parametric x-axis for the k-path (Γ → K → M → Γ)
        # k_path_length = np.arange(len(k_vec))
        # plt.figure(figsize=(10, 6))
        # plt.plot(k_path_length, Eplus, label='E+', color='blue')
        # plt.plot(k_path_length, Eminus, label='E-', color='red')

        # # Positions must be scalars
        # positions = [
        #     0, 
        #     N_k**2, 
        #     2 * N_k**2, 
        #     3 * N_k**2 - 1
        # ]
        # plt.xticks(
        #     positions,
        #     ['Γ', 'K', 'M', 'Γ'],
        #     fontsize=12
        # )

        # plt.xlabel('High-Symmetry Path')
        # plt.ylabel('Energy (eV)')
        # plt.legend()
        # plt.grid(True)
        # plt.show()