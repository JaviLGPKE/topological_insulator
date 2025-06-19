import numpy as np
from scipy import linalg
from sympy import LeviCivita
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
        # Parameters
        self.u_B = (6.63e-34)/(4 * np.pi * 9.11e-31)
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
        self.n_projections = len(self.uncoupled_states)
        self.U = self._coupled_unitary_transform()

    def _coupled_unitary_transform(self):
        """
        Transforms the 8x8 Hamiltonian from uncoupled orbital/spin basis 
        to coupled angular momentum basis, using the Clebsch-Gordan coefficients.

        Returns
        -------
        U : np.ndarray
            The unitary matrix. 
        """
        N = len(self.uncoupled_states)
        U = np.zeros((N, N), dtype=complex)
        U[0, 0] = 1  # |s, j=1/2, m_j=+1/2⟩ = |s↑⟩
        U[1, 1] = 1  # |s, j=1/2, m_j=-1/2⟩ = |s↓⟩
        # 1. Transformation: real p-orbitals → spherical harmonics
        # Basis: [p_x↑, p_x↓, p_y↑, p_y↓, p_z↑, p_z↓] → [|1,1>↑, |1,1>↓, |1,0>↑, |1,0>↓, |1,-1>↑, |1,-1>↓]
        U_orb = np.array([
            [-1/np.sqrt(2), -1j/np.sqrt(2), 0],  # |1,1>
            [ 1/np.sqrt(2), -1j/np.sqrt(2), 0],  # |1,-1>
            [ 0,            0,             1]    # |1,0>
        ], dtype=complex)
        # Expand to spin space (6x6 matrix)
        T_real2complex = np.kron(U_orb, np.eye(2))
        # NOTE: Original order was [|1,1>, |1,-1>, |1,0>] so we swap last two blocks
        P = np.eye(6)
        P = P[[0, 1, 4, 5, 2, 3]]  # New order: 0,1 stay; then 4,5; then 2,3
        T_real2complex = P @ T_real2complex
        # =============================================
        # Transformation: complex basis → coupled basis
        # =============================================
        # Basis: [|1,1>↑, |1,1>↓, |1,0>↑, |1,0>↓, |1,-1>↑, |1,-1>↓] 
        # → [j=3/2, m_j=3/2; j=3/2, m_j=1/2; j=3/2, m_j=-1/2; j=3/2, m_j=-3/2; j=1/2, m_j=1/2; j=1/2, m_j=-1/2]
        T_CG = np.zeros((6, 6), dtype=complex)
        sqrt1_3 = np.sqrt(1/3)
        sqrt2_3 = np.sqrt(2/3)
        # j = 3/2 states
        T_CG[0, 0] = 1                          # |3/2, 3/2⟩ = |1,1>↑
        T_CG[1, 1] = sqrt1_3; T_CG[1, 2] = sqrt2_3  # |3/2, 1/2⟩ = √(1/3)|1,1>↓ + √(2/3)|1,0>↑
        T_CG[2, 3] = sqrt2_3; T_CG[2, 4] = sqrt1_3  # |3/2,-1/2⟩ = √(2/3)|1,0>↓ + √(1/3)|1,-1>↑
        T_CG[3, 5] = 1                          # |3/2,-3/2⟩ = |1,-1>↓
        # j = 1/2 states
        T_CG[4, 1] = sqrt2_3; T_CG[4, 2] = -sqrt1_3  # |1/2, 1/2⟩ = √(2/3)|1,1>↓ - √(1/3)|1,0>↑
        T_CG[5, 3] = sqrt1_3; T_CG[5, 4] = -sqrt2_3  # |1/2,-1/2⟩ = √(1/3)|1,0>↓ - √(2/3)|1,-1>↑
        T_p = T_CG @ T_real2complex
        U[2:, 2:] = T_p
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

    def _sublattice_data(self, geometry:Geometry, location:str, idx:int):
        neighbour_idxs = geometry.get_neighbour_idxs(idx)
        dr_list_NN, _ = geometry.get_dr(location, idx, neighbour_idxs, type="list")
        dr_dict_NN, dm_dict_NN = geometry.get_dr(location, idx, neighbour_idxs, type="dict")
        directional_cosines_NN = geometry.bond_orientation(dr_list_NN)
        next_neighbour_idxs = geometry.get_next_neighbour_idxs(idx)
        dr_dict_NNN, dm_dict_NNN = geometry.get_dr(location, idx, next_neighbour_idxs, type="dict")
        # Hopping
        t_ij_dict = {}
        for neighbour_idx, cosines in zip(neighbour_idxs, directional_cosines_NN):
            eigenvalue_dict = self.slater_koster_hoppings(geometry, idx, neighbour_idx, cosines)
            H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = self.U.conj().T @ H_uncoupled @ self.U
            t_ij_dict[neighbour_idx] = H_coupled
        # Spin-Orbit Coupling
        s_ij_dict = {}
        for next_neighbour_idx in next_neighbour_idxs:
            eigenvalue_dict = self.spin_orbit_coupling(geometry, idx, next_neighbour_idx)
            H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = self.U.conj().T @ H_uncoupled @ self.U
            s_ij_dict[next_neighbour_idx] = H_coupled
        # TODO: Mean Field Decoupled Interaction
        u_ij_dict = {}
        # Zeeman-Splitting
        z_ij_dict = {}
        eigenvalue_dict = self.zeeman_splitting(geometry, idx)
        H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = self.U.conj().T @ H_uncoupled @ self.U
        z_ij_dict[idx] = H_coupled
        # Staggered Sublattice Potential
        m_ij_dict = {}
        eigenvalue_dict = self.staggered_sublattice_potential(geometry, idx)
        H_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = self.U.conj().T @ H_uncoupled @ self.U
        m_ij_dict[idx] = H_coupled
        return {
                "idx": idx,
                "NN_idxs": neighbour_idxs,
                "dr_dict_NN": dr_dict_NN,
                "dm_dict_NN": dm_dict_NN,
                "NNN_idxs": next_neighbour_idxs,
                "dr_dict_NNN": dr_dict_NNN,
                "dm_dict_NNN": dm_dict_NNN,
                "hopping_dict": t_ij_dict,
                "spin_orbit_coupling_dict": s_ij_dict,
                "interaction_dict": u_ij_dict,
                "zeeman_splitting_dict": z_ij_dict,
                "staggered_potential_dict": m_ij_dict     
        }

    def slater_koster_hoppings(self, geometry, site_i, site_j, cosines):
        label_i, label_j = geometry.get_label(site_i), geometry.get_label(site_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        nn_parser = eigenvalue_parser.value["nn_hopping"]
        t_ss = nn_parser[label_j]["t_ss_sigma"]
        t_sp = nn_parser[label_j]["t_sp_sigma"]
        t_pp_sigma = nn_parser[label_j]["t_pp_sigma"]
        t_pp_pi = nn_parser[label_j]["t_pp_pi"]
        l, m = (cosines[0], cosines[1])
        n = cosines[2] if len(cosines) == 3 else 0
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
                outer_product = f"|{alpha}><{beta}|"
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
                    H_t += p_cosines[d] * t_sp
                    if outer_product[1] != "s":
                        H_t *= -1
                # p-p
                elif alpha.startswith('p') and beta.startswith('p'):
                    i = self.direction_index[alpha.split('_')[1]]
                    j = self.direction_index[beta.split('_')[1]]
                    H_t += pp_matrix[i][j]
                else: 
                    raise ValueError(f"Not Implemented!")
                # Spin
                for sigma_1 in self.spin_dict.values():
                    for sigma_2 in self.spin_dict.values():
                            outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                            eigenvalue_dict[outer_product] = H_t if sigma_1 == sigma_2 else 0
        return eigenvalue_dict

    def spin_orbit_coupling(self, geometry:Geometry, site_i, site_j):
        label_i, label_j = geometry.get_label(site_i), geometry.get_label(site_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["SO_coupling"][label_j]
        lambda_ss = so_parser["lambda_ss"]
        lambda_sp = so_parser["lambda_sp"]
        lambda_pp = so_parser["lambda_pp"]
        v_ij = geometry.get_chirality(site_i, site_j)
        coupling_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                        H_so = 0
                        # s-s Kane coupling
                        if alpha == "s" and beta == "s":
                            # NOTE: <s|L·S|s> = 0
                            pauli_matrix = self.pauli_matrix_dict[2]
                            H_so += 1j * lambda_ss * pauli_matrix[n, m]
                        # s-p or p-s Kane coupling
                        elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                            p_orb = alpha if alpha.startswith('p') else beta
                            d = self.direction_index[p_orb.split('_')[1]]
                            pauli_matrix = self.pauli_matrix_dict[d]
                            H_so += 1j * lambda_sp * pauli_matrix[n, m]
                            #NOTE: s are symmetric, p orbitals are antisymmetric -> Under spatial inversion:
                            # s(-r) = s(r) and p(-r) = -p(r), hence <s|H|x> = -<x|H|s>
                            if outer_product[1] != "s":
                                H_so *= -1
                        # p-p Chadi coupling
                        elif alpha.startswith('p') and beta.startswith('p'):
                            i = self.direction_index[alpha.split('_')[1]]
                            j = self.direction_index[beta.split('_')[1]]
                            k = (set(self.direction_index.values()) - {i, j}).pop()
                            eps_ijk = LeviCivita(i, j, k)
                            sigma_k = self.pauli_matrix_dict[k]
                            H_so += 1j * lambda_pp * eps_ijk * sigma_k[n, m]
                        else: 
                            raise ValueError(f"Not Implemented!")
                        coupling_dict[outer_product] = v_ij * H_so
        return coupling_dict
    
    def mean_field_interaction(self, site_i):
        # TODO:
        return {}

    def zeeman_splitting(self, geometry:Geometry, site_i):
        # TODO: coupling between spin and orbital
        coupling_dict = {}
        B = self.cell_parser.field.magnetic.value
        u_B = self.u_B
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                        H_z = 0
                        if alpha == "s":
                            pauli_matrix = self.pauli_matrix_dict[2]
                            H_z += 1/2 * u_B  * B * pauli_matrix[n, m]
                        coupling_dict[outer_product] = H_z
        return coupling_dict

    def staggered_sublattice_potential(self, geometry: Geometry, site_i):
        coupling_dict = {}
        label_i = geometry.get_label(site_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        m_parser = eigenvalue_parser.value["onsite_energy"][label_i]
        E_s = m_parser["E_s"]
        E_p = m_parser["E_p"]
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for alpha in self.orbitals:
                outer_product = f"|{alpha},{sigma_1}><{alpha},{sigma_1}|"
                H_z = 0
                if alpha == "s":
                    H_z += E_s
                elif alpha.startswith('p'):
                    H_z += E_p
                else: 
                    raise ValueError(f"Not Implemented!")
                coupling_dict[outer_product] = H_z
        return coupling_dict

    def _uncoupled_eigenvalue_matrix(self, eigenvalue_dict:dict):
        uncoupled_states = self.uncoupled_states
        N = len(uncoupled_states)
        H_uncoupled = np.zeros((N, N), dtype=complex)
        for i, (alpha, sigma_1) in enumerate(uncoupled_states):
            for j, (beta, sigma_2) in enumerate(uncoupled_states):
                outer_product = f"|{alpha},{sigma_1}><{beta},{sigma_2}|"
                try:
                    E_ij = eigenvalue_dict[outer_product]
                except: 
                    E_ij = 0
                H_uncoupled[i, j] = E_ij
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
