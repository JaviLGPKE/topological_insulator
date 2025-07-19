import numpy as np
from scipy import linalg
from sympy.physics.quantum.cg import CG, wigner_3j
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
        self.orbital_states = [
            (orb, sigma) 
            for orb in self.orbitals 
            for sigma in self.spin_dict.values()
        ]
        self.uncoupled_states = [
            ((0, 0), (1/2, +1/2)), ((0, 0), (1/2, -1/2)), 
            ((1, +1), ((1/2, +1/2))), ((1, +1), (1/2, -1/2)),
            ((1, -1), ((1/2, +1/2))), ((1, -1), (1/2, -1/2)),
            ((1, 0), ((1/2, +1/2))), ((1, 0), (1/2, -1/2))
        ]
        self.coupled_states = [
            (1/2, +1/2), (1/2, -1/2), (1/2, +1/2), (1/2, -1/2), 
            (3/2, +3/2), (3/2, +1/2), (3/2, -1/2), (3/2, -3/2)
        ]
        self.C = self._coupled_unitary_transform()
        self.T = self._harmonic_unitary_transform()
        # Parity
        self.O = self._time_reversal_operator()

    def _coupled_unitary_transform(self):
        """
        Transforms the 8x8 Hamiltonian from uncoupled angular momentum/spin basis 
        to coupled angular momentum basis, using the Clebsch-Gordan (CG) coefficients.

        Returns
        -------
        U : np.ndarray
            The Clebsch-Gordan unitary matrix. 
        """
        # NOTE: follows notebook format
        M, N = len(self.coupled_states), len(self.uncoupled_states)
        C = np.zeros((N, M), dtype=float)
        # l = 0
        C[0, 0] = CG(0, 0, 1/2, +1/2, 1/2, +1/2).doit()
        C[1, 1] = CG(0, 0, 1/2, -1/2, 1/2, -1/2).doit()
        # l = 1
        C[2, 3] = CG(1, +1, 1/2, -1/2, 1/2, +1/2).doit()
        C[2, 6] = CG(1, 0, 1/2, +1/2, 1/2, +1/2).doit()
        C[3, 4] = CG(1, -1, 1/2, +1/2, 1/2, -1/2).doit()
        C[3, 7] = CG(1, 0, 1/2, -1/2, 1/2, -1/2).doit()
        C[4, 2] = CG(1, +1, 1/2, +1/2, 3/2, +3/2).doit()
        C[5, 3] = CG(1, +1, 1/2, -1/2, 3/2, +1/2).doit()
        C[5, 6] = CG(1, 0, 1/2, +1/2, 3/2, +1/2).doit()
        C[6, 4] = CG(1, -1, 1/2, +1/2, 3/2, -1/2).doit()
        C[6, 7] = CG(1, 0, 1/2, -1/2, 3/2, -1/2).doit()
        C[7, 5] = CG(1, -1, 1/2, -1/2, 3/2, -3/2).doit()
        return C

    def _harmonic_unitary_transform(self):
        """
        Transforms the 8x8 Hamiltonian from cartesian orbital/spin basis 
        to uncoupled angular momentum basis, using Spherical Harmonics.

        Returns
        -------
        J : np.ndarray
            The Spherical Harmonic unitary matrix. 
        """
        M, N = len(self.uncoupled_states), len(self.orbital_states)
        T = np.zeros((N, M), dtype=complex)
        inv_sqrt_2 = 1/np.sqrt(2)
        # s-orbitals
        T[0, 0] = 1
        T[1, 1] = 1
        # p-orbitals
        T[2, 2] = -1 * inv_sqrt_2
        T[2, 4] = 1j * inv_sqrt_2
        T[3, 3] = -1 * inv_sqrt_2
        T[3, 5] = 1j * inv_sqrt_2
        T[4, 2] = 1 * inv_sqrt_2
        T[4, 4] = 1j * inv_sqrt_2
        T[5, 3] = 1 * inv_sqrt_2
        T[5, 5] = 1j * inv_sqrt_2
        T[6, 6] = 1
        T[7, 7] = 1
        return T

    def _time_reversal_operator(self):
        """
        The Time-Reversal (TR) operator. NOTE: for usage always apply the complex conjugate to
        the operator or state the TR operator acts upon.

        Returns
        -------
        O : np.ndarray
            The Time-Reversal matrix. 
        """
        eigenvalue_dict = {}
        S_y = self.pauli_matrix_dict[self.direction_index["y"]]
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    outer_product = f"|{alpha}, {sigma_1}><{alpha}, {sigma_2}|"
                    eigenvalue_dict[outer_product] = -1j * S_y[n, m]
        O_uncoupled = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        C = self.C
        T = self.T
        P = C @ T
        P_dagger = T.conj().T @ C.conj().T
        O_coupled = P @ O_uncoupled @ P_dagger.conj()
        O_sublattice = np.identity(n=len(self.delta_vectors))
        O = np.kron(O_sublattice, O_coupled)
        return O

    @abstractmethod
    def build_hamiltonian(self, geometry:Geometry) -> None:
        """
        Must build the necessary sublattice data for the 'calculate_eigenvalues' method.
        """
        self.sublattice_data_dict = None
        self.sublattice_connectivity = None
        self.H = None
        raise NotImplementedError("'build_hamiltonian' method not implemented!")

    def _sublattice_data(self, geometry:Geometry, location:str, idx_i:int):
        C = self.C # Clebsch-Gordan Transformation Matrix
        T = self.T # Cartesian to Angular Momentum Unitary Transform
        P = C @ T
        P_dagger = T.conj().T @ C.conj().T
        neighbour_idxs = geometry.get_neighbour_idxs(idx_i)
        dr_list_NN, _ = geometry.get_dr(location, idx_i, neighbour_idxs, type="list")
        dr_dict_NN, dm_dict_NN = geometry.get_dr(location, idx_i, neighbour_idxs, type="dict")
        directional_cosines_NN = geometry.bond_orientation(dr_list_NN)
        next_neighbour_idxs = geometry.get_next_neighbour_idxs(idx_i)
        dr_dict_NNN, dm_dict_NNN = geometry.get_dr(location, idx_i, next_neighbour_idxs, type="dict")
        # Hopping
        t_ij_dict = {}
        for idx_j, cosines in zip(neighbour_idxs, directional_cosines_NN):
            eigenvalue_dict = self.slater_koster_hoppings(geometry, idx_i, idx_j, cosines)
            H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = P @ H_cartesian @ P_dagger
            t_ij_dict[idx_j] = self.hopping_anisotropy(geometry, idx_i, idx_j, H_coupled)
        # Kane-Mele Spin-Orbit Coupling
        s_ij_dict = {}
        for idx_j in next_neighbour_idxs:
            eigenvalue_dict = self.kane_mele_coupling(geometry, idx_i, idx_j)
            H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = P @ H_cartesian @ P_dagger
            s_ij_dict[idx_j] = H_coupled
        # Chaid Spin-Orbit Coupling
        c_ij_dict = {}
        eigenvalue_dict = self.chadi_coupling(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = P @ H_cartesian @ P_dagger
        c_ij_dict[idx_i] = H_coupled
        # Mean Field Decoupled Interaction
        u_ij_dict = {}
        eigenvalue_dict = self.mean_field_interaction(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = P @ H_cartesian @ P_dagger
        u_ij_dict[idx_i] = self.interaction_anisotropy(geometry, idx_i, H_coupled)
        # Zeeman-Splitting
        z_ij_dict = {}
        eigenvalue_dict = self.zeeman_splitting(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = P @ H_cartesian @ P_dagger
        z_ij_dict[idx_i] = H_coupled
        # Staggered Sublattice Potential
        m_ij_dict = {}
        eigenvalue_dict = self.staggered_sublattice_potential(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = P @ H_cartesian @ P_dagger
        m_ij_dict[idx_i] = H_coupled
        return {
                "idx": idx_i,
                "NN_idxs": neighbour_idxs,
                "dr_dict_NN": dr_dict_NN,
                "dm_dict_NN": dm_dict_NN,
                "NNN_idxs": next_neighbour_idxs,
                "dr_dict_NNN": dr_dict_NNN,
                "dm_dict_NNN": dm_dict_NNN,
                "hopping_dict": t_ij_dict,
                "kane_mele_coupling_dict": s_ij_dict,
                "chadi_coupling_dict": c_ij_dict,
                "mean_field_interaction_dict": u_ij_dict,
                "zeeman_splitting_dict": z_ij_dict,
                "staggered_potential_dict": m_ij_dict     
        }

    def slater_koster_hoppings(self, geometry:Geometry, idx_i, idx_j, cosines):
        label_i, label_j = geometry.get_label(idx_i), geometry.get_label(idx_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        nn_parser = eigenvalue_parser.value["nn_hopping"][label_j]
        t_ss = nn_parser["t_ss_sigma"]
        t_sp = nn_parser["t_sp_sigma"]
        t_pp_sigma = nn_parser["t_pp_sigma"]
        t_pp_pi = nn_parser["t_pp_pi"]
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
                            outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                            eigenvalue_dict[outer_product] = H_t if sigma_1 == sigma_2 else 0
        return eigenvalue_dict

    def hopping_anisotropy(self, geometry:Geometry, idx_i, idx_j, H):
        label_i, label_j = geometry.get_label(idx_i), geometry.get_label(idx_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        nn_parser = eigenvalue_parser.value["nn_hopping"][label_j]
        d_heavy = nn_parser["delta_heavy"]
        d_light = nn_parser["delta_light"]
        for i, (j, m_j) in enumerate(self.coupled_states):
            for n, (k, m_k) in enumerate(self.coupled_states):
                if j == 3/2 and k == 3/2:
                    if m_j == 3/2 and m_k == 3/2:
                        H[i, n] += d_heavy
                    elif m_j == -3/2 and m_k == -3/2:
                        H[i, n] += d_heavy
                    elif m_j == 1/2 and m_k == 1/2:
                        H[i, n] -= d_light
                    elif m_j == -1/2 and m_k == -1/2:
                        H[i, n] -= d_light
                    else: 
                        continue
        return H

    def kane_mele_coupling(self, geometry:Geometry, idx_i, idx_j):
        # FIXME: Check if there is an error for p-orbital implementation
        label_i, label_j = geometry.get_label(idx_i), geometry.get_label(idx_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["kane_mele_soc"][label_j]
        lambda_ss = so_parser["lambda_ss"]
        lambda_sp = so_parser["lambda_sp"]
        lambda_pp = so_parser["lambda_pp"]
        v_ij = geometry.get_chirality(idx_i, idx_j)
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_so = 0
                        # s-s Kane coupling
                        if alpha == "s" and beta == "s":
                            # NOTE: <s|L·S|s> = 0
                            pauli_matrix = self.pauli_matrix_dict[2]
                            H_so += 1j * lambda_ss *  v_ij * pauli_matrix[n, m]
                        # s-p or p-s Kane coupling
                        elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                            # p_orb = alpha if alpha.startswith('p') else beta
                            # d = self.direction_index[p_orb.split('_')[1]]
                            # pauli_matrix = self.pauli_matrix_dict[d]
                            # H_so += 1j * lambda_sp * v_ij * pauli_matrix[n, m]
                            # #NOTE: s are symmetric, p orbitals are antisymmetric -> Under spatial inversion:
                            # # s(-r) = s(r) and p(-r) = -p(r), hence <s|H|x> = -<x|H|s>
                            # if outer_product[1] != "s":
                            #     H_so *= -1
                            pass
                        # p-p Chadi coupling
                        elif alpha.startswith('p') and beta.startswith('p'):
                            pass
                        else: 
                            raise ValueError(f"Not Implemented!")
                        eigenvalue_dict[outer_product] = H_so
        return eigenvalue_dict

    def chadi_coupling(self, geometry:Geometry, idx_i):
        label_i = geometry.get_label(idx_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["chadi_soc"][label_i]
        lambda_ss = so_parser["lambda_ss"]
        lambda_sp = so_parser["lambda_sp"]
        lambda_pp = so_parser["lambda_pp"]
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_so = 0
                        if alpha == "s" and beta == "s":
                            pass
                        elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                            pass
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
                        eigenvalue_dict[outer_product] = H_so
        return eigenvalue_dict

    def mean_field_interaction(self, geometry:Geometry, idx_i):
        label_i,  = geometry.get_label(idx_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        int_parser = eigenvalue_parser.value["interaction"][label_i]
        U = int_parser["U"]
        n_s_up, n_s_down = int_parser["n_s_up"], int_parser["n_s_down"]
        n_p_up, n_p_down = int_parser["n_p_up"], int_parser["n_p_down"]
        # Constant Shift
        E_int = 0
        for alpha in self.orbitals:
            if alpha == "s":
                E_int += U * n_s_up * n_s_down
            elif alpha.startswith('p'):
                E_int += U * n_p_up * n_p_down
        sigma_x = self.pauli_matrix_dict[0]
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    outer_product = f"|{alpha}, {sigma_1}><{alpha}, {sigma_1}|"
                    H_int = 0
                    if sigma_1 != sigma_2:
                        continue
                    if alpha == "s":
                        if sigma_1 == "+":
                            H_int += U * n_p_down
                        elif sigma_1 == "-":
                            H_int += U * n_p_up
                        else:
                            ValueError("help")
                    elif alpha.startswith('p'):
                        if sigma_1 == "+":
                            H_int += U * n_p_down
                        elif sigma_1 == "-":
                            H_int += U * n_p_up
                        else:
                            ValueError("help")
                    eigenvalue_dict[outer_product] = (H_int) - E_int
        return eigenvalue_dict
    
    def interaction_anisotropy(self, geometry:Geometry, idx_i, H_coupled):
        for i, (j, m_j) in enumerate(self.coupled_states):
            if j != 3/2:
                H_coupled[i, i] = 0
        return H_coupled

    def zeeman_splitting(self, geometry:Geometry, site_i):
        # TODO: coupling between spin and orbital i.e. m_l
        eigenvalue_dict = {}
        B = self.cell_parser.field.magnetic.value
        u_B = self.u_B
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_z = 0
                        if alpha == "s":
                            pauli_matrix = self.pauli_matrix_dict[2]
                            H_z += 1/2 * u_B  * B * pauli_matrix[n, m]
                        eigenvalue_dict[outer_product] = H_z
        return eigenvalue_dict

    def staggered_sublattice_potential(self, geometry: Geometry, site_i):
        eigenvalue_dict = {}
        label_i = geometry.get_label(site_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        m_parser = eigenvalue_parser.value["onsite_energy"][label_i]
        E_s = m_parser["E_s"]
        E_p = m_parser["E_p"]
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for alpha in self.orbitals:
                outer_product = f"|{alpha}, {sigma_1}><{alpha}, {sigma_1}|"
                H_z = 0
                if alpha == "s":
                    H_z += E_s
                elif alpha.startswith('p'):
                    H_z += E_p
                else: 
                    raise ValueError(f"Not Implemented!")
                eigenvalue_dict[outer_product] = H_z
        return eigenvalue_dict

    def _uncoupled_eigenvalue_matrix(self, eigenvalue_dict:dict):
        orbital_states = self.orbital_states
        N = len(orbital_states)
        H_uncoupled = np.zeros((N, N), dtype=complex)
        for i, (alpha, sigma_1) in enumerate(orbital_states):
            for j, (beta, sigma_2) in enumerate(orbital_states):
                outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
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
            self.H_k_dict = None
            self.E_k_dict, self.U_k_dict = None, None
        raise NotImplementedError("'solve_eigenvalues' method not implemented")

    def _solve_eigenvalues(self, H):
        E, U = linalg.eigh(H, lower=True, check_finite=False, driver="evr")
        return E, U

    @abstractmethod
    def plot_dispersion(self, geometry: Geometry):
        raise NotImplementedError("Implement dispersion plot method!")
