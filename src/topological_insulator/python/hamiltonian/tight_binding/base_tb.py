import numpy as np
from scipy import linalg
from sympy import LeviCivita
from sympy.physics.quantum.cg import CG
import re
from matplotlib import pyplot as plt
from abc import abstractmethod

from ..notation import Notation
from ...model_options import ModelOptions
from ...cell_parser import CellParser
from ...geometry import Geometry

from IPython import embed

class TightBinding(Notation):
    """
    Tight-Binding approximation Hamiltonian that can include, nearest neighbour hopping, 
    spin-orbit coupling interaction and Coulomb repulsive interaction terms.
    """

    def __init__(self, model_options:ModelOptions, cell_parser: CellParser):
        super().__init__()
        # Arguments
        self.model_options = model_options
        self.cell_parser = cell_parser
        # Sublattice
        self.sublattice_data_dict = {}
        self.basis_vectors = np.array(cell_parser.geometry.lattice_vectors.value)
        self.delta_vectors = np.array(cell_parser.geometry.delta_vectors.value)
        # Clebsch-Gordan Coefficients
        self.uncoupled_states = [
            (orb, sigma) 
            for orb in self.orbitals 
            for sigma in self.spin_dict.values()
        ]
        self.coupled_states =  [
            (0.5, -0.5), (0.5, 0.5),
            (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)
        ]
        self.n_projections = len(self.coupled_states)
        self._clebsch_gordan()
        self.U = self._coupled_unitary_transform()

    def _clebsch_gordan(self):
        self.CG_coefficients = {}
        j_2 = 1/2
        m_2 = np.arange(-j_2, j_2 + 1, 1)
        for j_1 in [0, 1]:          
            m_1 = np.arange(-j_1, j_1 + 1, 1)
            j_3 = j_1 + j_2
            m_3 = np.arange(-j_3, j_3 + 1, 1)
            for i, m_j in enumerate(m_3):
                for m_l in m_1:
                    for m_s in m_2:
                        state = f"|{j_1},{m_l};{j_2},{m_s}>"
                        if (m_l + m_s) != m_j:
                            continue
                        self.CG_coefficients[state] = CG(j_1, m_l, j_2, m_s, j_3, m_j).doit()
    
    def _coupled_unitary_transform(self):
        """
        Transforms the 8x8 Hamiltonian (s + p orbitals with spin-1/2)
        into the coupled |j, m_j> basis. Returns a transformed 6x6 Hamiltonian.

        Returns
        -------
        U : np.ndarray
            The unitary matrix that transforms the Hamiltonian from a 8x8 to a 6x6 matrix. 
              total-angular-momentum coupled basis |j, m_j>.
        """
        # Unitary Transform
        M, N = len(self.uncoupled_states), len(self.coupled_states)
        U = np.zeros(shape=(M, N), dtype=complex)
        for i, (bra_state, bra_CG) in enumerate(self.CG_coefficients.items()):
            bra_l   = self.get_quantum_number(bra_state, pos=0)
            bra_m_l = self.get_quantum_number(bra_state, pos=1)
            bra_m_s = self.get_quantum_number(bra_state, pos=3)
            bra_j = bra_l + 1/2
            bra_m_j = bra_m_l + bra_m_s
            bra_orbitals = self.l_to_orbitals(bra_l, bra_m_l)
            for j, (ket_j, ket_m_j) in enumerate(self.coupled_states):
                # ket_state = f"|{ket_j},{ket_m_j}>"
                if (bra_j == ket_j) and (bra_m_j == ket_m_j): 
                    for bra_orb, bra_coeff in bra_orbitals.items():
                        U[i, j] += bra_coeff * bra_CG
        return U

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
        dr_list, dm_list = geometry.get_dr(location, idx, neighbour_idxs, type="list")
        dr_dict, dm_dict = geometry.get_dr(location, idx, neighbour_idxs, type="dict")
        directional_cosines = geometry.bond_orientation(dr_list)
        t_ij_dict = self.get_hamiltonian_submatrix(
            geometry, idx, neighbour_idxs, directional_cosines, eigenvalue_type="hopping")
        s_ij_dict = self.get_hamiltonian_submatrix(
            geometry, idx, neighbour_idxs, directional_cosines, eigenvalue_type="spin_orbit_coupling")
        return {
                "idx": idx,
                "neighbour_idxs": neighbour_idxs,
                "dr_dict": dr_dict,
                "dm_dict": dm_dict,
                "hopping_dict": t_ij_dict,
                "spin_orbit_coupling_dict": s_ij_dict,
                "interaction_dict": None
        }

    def get_hamiltonian_submatrix(self, geometry:Geometry, idx, neighbour_idxs, directional_cosines, eigenvalue_type="hppping"):
        label_i = geometry.get_label(idx)
        H_i = {}
        if eigenvalue_type == "hopping":
            for neighbour_idx, cosines in zip(neighbour_idxs, directional_cosines):
                j = neighbour_idx
                label_j = geometry.get_label(j)
                eigenvalue_dict = self._slater_koster_hoppings(label_i, label_j, cosines)
                H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
                H_coupled = self.U.conj().T @ H_uncoupled @ self.U 
                H_i[j] = H_coupled
        elif eigenvalue_type == "spin_orbit_coupling":
            i = idx
            eigenvalue_dict = self._spin_orbit_coupling(label_i)
            H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = self.U.conj().T @ H_uncoupled @ self.U 
            H_i[i] = H_coupled
        else:
            raise ValueError(f"'{eigenvalue_type}' not implemented!")  
        return H_i    

    def _slater_koster_hoppings(self, label_i, label_j, cosines):
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        nn_parser = eigenvalue_parser.value["nn_hopping"]
        t_ss = nn_parser[label_j]["t_ss_sigma"]
        t_sp = nn_parser[label_j]["t_sp_sigma"]
        t_pp_sigma = nn_parser[label_j]["t_pp_sigma"]
        t_pp_pi = nn_parser[label_j]["t_pp_pi"]
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
        # Hopping Eigenvalues
        eigenvalue_dict = {}
        for alpha in self.orbitals:
            for beta in self.orbitals:
                key = f"|{alpha}><{beta}|"
                H_t = 0
                # s-s
                if alpha == beta == 's':
                    H_t += t_ss
                # s-p or p-s
                elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                    p_orb = alpha if alpha.startswith('p') else beta
                    d = self.direction_index[p_orb.split('_')[1]]
                    #NOTE: s are symmetric, p orbitals are antisymmetric -> Under spatial inversion:
                    # s(-r) = s(r) and p(-r) = -p(r), hence <s|H|x> = -<x|H|s>
                    t = p_cosines[d] * t_sp
                    if key[1] != "s":
                        t *= -1
                    H_t += t
                # p-p
                elif alpha.startswith('p') and beta.startswith('p'):
                    i = self.direction_index[alpha.split('_')[1]]
                    j = self.direction_index[beta.split('_')[1]]
                    H_t += pp_matrix[i][j]
                # Spin
                for sigma_1 in self.spin_dict.values():
                    for sigma_2 in self.spin_dict.values():
                        outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                        eigenvalue_dict[outer_product] = H_t
        return eigenvalue_dict

    def _spin_orbit_coupling(self, label_i):
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["SO_coupling"][label_i]
        lambda_SO = so_parser["lambda_pp"]
        coupling_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                        H_SO = 0
                        # p-p
                        if alpha.startswith('p') and beta.startswith('p'):
                            i = self.direction_index[alpha.split('_')[1]]
                            j = self.direction_index[beta.split('_')[1]]
                            k = (set(self.direction_index.values()) - {i, j}).pop()
                            eps_ijk = LeviCivita(i, j, k)
                            sigma_k = self.pauli_matrix_dict[k]
                            H_SO += 1j * lambda_SO * eps_ijk * sigma_k[n, m]
                        coupling_dict[outer_product] = H_SO  
        return coupling_dict

    def _uncoupled_eigenvalue_matrix(self, eigenvalue_dict:dict):
        uncoupled_states = self.uncoupled_states
        N = len(uncoupled_states)
        H_uncoupled = np.zeros((N, N), dtype=complex)
        for i, (alpha, sigma_1) in enumerate(uncoupled_states):
            for j, (beta, sigma_2) in enumerate(uncoupled_states):
                outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                H_uncoupled[i, j] = eigenvalue_dict[outer_product]
        return H_uncoupled

    @abstractmethod
    def solve_eigenvalues(self, geometry:Geometry, H_type:str):
        """
        Must calculate the necessary eigenvalues depending on the requested 
        Hamiltonian type.
        """
        if H_type == "real_space":
            self.E = None
        elif H_type == "reciprocal_space":
            self.E_k_dict, self.U_k_dict = None, None
        raise NotImplementedError("'solve_eigenvalues' method not implemented")

    def _solve_eigenvalues(self, H):
        E, U = linalg.eigh(H, lower=True, check_finite=False, driver="evr")
        return E, U

    @abstractmethod
    def plot_dispersion(self, geometry: Geometry):
        raise NotImplementedError("Implement dispersion plot method!")

    def plot_band_structure(self, geometry: Geometry):
        raise NotImplementedError("Implement band structure plot method!")
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