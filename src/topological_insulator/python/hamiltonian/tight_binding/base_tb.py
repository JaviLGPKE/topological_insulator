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
        self.band_structure_data = None
        # Parameters
        self.u_B = (6.63e-34)/(4 * np.pi * 9.11e-31)
        # Sublattice
        self.sublattice_data_dict = {}
        self.basis_vectors = np.array(cell_parser.geometry.lattice_vectors.value)
        self.delta_vectors = np.array(cell_parser.geometry.delta_vectors.value)
        # Clebsch-Gordan Coefficients
        self.orbital_mapping = {
            "s": [0, 1],
            "p": [2, 8]
        }
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
        self.A = self._harmonic_unitary_transform()
        # Parity
        self.O = self._time_reversal_operator()
        # Minimisation of Free Energy
        self.E_0 = self.get_E_0(cell_parser)

    def _coupled_unitary_transform(self):
        """
        Transforms the 8x8 Hamiltonian from uncoupled angular momentum/spin basis 
        to coupled angular momentum basis, using the Clebsch-Gordan (CG) coefficients.

        Returns
        -------
        U : np.ndarray
            The Clebsch-Gordan unitary matrix. 
        """
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
        A = np.zeros((N, M), dtype=complex)
        inv_sqrt_2 = 1/np.sqrt(2)
        # s-orbitals
        A[0, 0] = 1
        A[1, 1] = 1
        # p-orbitals
        A[2, 2] = -1 * inv_sqrt_2
        A[2, 4] = 1j * inv_sqrt_2
        A[3, 3] = -1 * inv_sqrt_2
        A[3, 5] = 1j * inv_sqrt_2
        A[4, 2] = 1 * inv_sqrt_2
        A[4, 4] = 1j * inv_sqrt_2
        A[5, 3] = 1 * inv_sqrt_2
        A[5, 5] = 1j * inv_sqrt_2
        A[6, 6] = 1
        A[7, 7] = 1
        return A

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
        A = self.A
        M = C @ A
        M_dagger = A.conj().T @ C.conj().T
        O_coupled = M @ O_uncoupled @ M_dagger
        O_sublattice = np.identity(n=len(self.delta_vectors))
        O = np.kron(O_sublattice, O_coupled)
        return O

    def get_E_0(self, cell_parser):
        g = cell_parser.geometry
        n_subs = len(g.delta_vectors.value)
        subs = self.sublattice_labels[:n_subs]
        E_0 = 0
        for i, label_i in enumerate(subs):
            eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
            int_parser = eigenvalue_parser.value["interaction"][label_i]
            U_s = int_parser["U_s"]
            U_p = int_parser["U_p"]
            n_s_up, n_s_down = int_parser["n_s_up"], int_parser["n_s_down"]
            n_px_up, n_px_down = int_parser["n_px_up"], int_parser["n_px_down"]
            n_py_up, n_py_down = int_parser["n_py_up"], int_parser["n_py_down"]
            n_pz_up, n_pz_down = int_parser["n_pz_up"], int_parser["n_pz_down"]
            E_0 += (
                U_s * n_s_up * n_s_down +
                U_p * (
                (n_px_up * n_px_down) + 
                (n_py_up * n_py_down) + 
                (n_pz_up * n_pz_down)
                )
            )
        return E_0

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
        A = self.A # Cartesian to Angular Momentum Unitary Transform
        M = C @ A
        M_dagger = A.conj().T @ C.conj().T
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
            H_coupled = M @ H_cartesian @ M_dagger
            t_ij_dict[idx_j] = H_coupled
        # Kane-Mele Spin-Orbit Coupling
        s_ij_dict = {}
        for idx_j in next_neighbour_idxs:
            eigenvalue_dict = self.kane_mele_coupling(geometry, idx_i, idx_j)
            H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
            H_coupled = M @ H_cartesian @ M_dagger
            s_ij_dict[idx_j] = H_coupled
        # Chadi Spin-Orbit Coupling
        c_ij_dict = {}
        eigenvalue_dict = self.chadi_coupling(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = M @ H_cartesian @ M_dagger
        c_ij_dict[idx_i] = H_coupled
        # Mean Field Decoupled Interaction
        u_ij_dict = {}
        eigenvalue_dict = self.mean_field_interaction(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = M @ H_cartesian @ M_dagger
        u_ij_dict[idx_i] = H_coupled
        # Zeeman-Splitting
        z_ij_dict = {}
        eigenvalue_dict = self.zeeman_splitting(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = M @ H_cartesian @ M_dagger
        z_ij_dict[idx_i] = H_coupled
        # Staggered Sublattice Potential
        m_ij_dict = {}
        eigenvalue_dict = self.onsite_energy(geometry, idx_i)
        H_cartesian = self._uncoupled_eigenvalue_matrix(eigenvalue_dict)
        H_coupled = M @ H_cartesian @ M_dagger
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
        t_ss_sigma = nn_parser["t_ss_sigma"]
        t_sp_sigma = nn_parser["t_sp_sigma"]
        t_pp_sigma = nn_parser["t_pp_sigma"]
        t_pp_pi = nn_parser["t_pp_pi"]
        l, m = (cosines[0], cosines[1])
        n = geometry.buckling_cosine if geometry.n_sublattices == 2 else 0
        p_cosines = [l, m, n]
        # Hopping Eigenvalues
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                if sigma_1 != sigma_2:
                    continue
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_t = 0
                        # s-s
                        if alpha == beta == 's':
                            H_t += t_ss_sigma
                        # s-p
                        elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                            p_orb = alpha if alpha.startswith('p') else beta
                            d = self.direction_index[p_orb.split('_')[1]]
                            H_t += p_cosines[d] * t_sp_sigma
                        # p-p
                        elif alpha.startswith('p') and beta.startswith('p'):
                            i_dir = alpha.split('_')[1]
                            j_dir = beta.split('_')[1]
                            i = self.direction_index[i_dir]
                            j = self.direction_index[j_dir]
                            if alpha == beta:
                                H_t = p_cosines[i]**2 * t_pp_sigma + (1 - p_cosines[i]**2) * t_pp_pi
                            else:
                                H_t = p_cosines[i] * p_cosines[j] * (t_pp_sigma - t_pp_pi)
                        else: 
                            raise ValueError(f"Not Implemented!")
                        eigenvalue_dict[outer_product] = H_t
        return eigenvalue_dict

    def kane_mele_coupling(self, geometry:Geometry, idx_i, idx_j):
        label_i, label_j = geometry.get_label(idx_i), geometry.get_label(idx_j)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["kane_mele_soc"][label_j]
        lambda_ss = so_parser["lambda_ss"]
        lambda_sp = so_parser["lambda_sp"]
        lambda_pp = so_parser["lambda_pp"]
        sigma_z = self.pauli_matrix_dict[2]
        v_ij = geometry.get_chirality(idx_i, idx_j)
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_km = 0
                        # s-s
                        if alpha == "s" and beta == "s":
                            # NOTE: <s|L·S|s> = 0
                            H_km += 1j * lambda_ss *  v_ij * sigma_z[n, m]
                        # s-p or p-s
                        elif (alpha == 's' and beta.startswith('p')) or (beta == 's' and alpha.startswith('p')):
                            # no SO coupling
                            pass
                        # p-p
                        elif alpha.startswith('p') and beta.startswith('p'):
                            if alpha == "p_z" and beta == "p_z":
                                H_km = 1j * lambda_pp * v_ij * sigma_z[n, m]
                            else:
                                pass
                        else: 
                            raise ValueError(f"Not Implemented!")
                        eigenvalue_dict[outer_product] = H_km
        return eigenvalue_dict

    def chadi_coupling(self, geometry:Geometry, idx_i):
        label_i = geometry.get_label(idx_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        so_parser = eigenvalue_parser.value["chadi_soc"][label_i]
        lambda_ss = so_parser["Delta_ss"]
        lambda_sp = so_parser["Delta_sp"]
        lambda_pp = so_parser["Delta_pp"]
        eigenvalue_dict = {}
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for m, sigma_2 in enumerate(self.spin_dict.values()):
                for alpha in self.orbitals:
                    for beta in self.orbitals:
                        outer_product = f"|{alpha}, {sigma_1}><{beta}, {sigma_2}|"
                        H_c = 0
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
                            H_c += 1j * lambda_pp * eps_ijk * sigma_k[n, m]
                        else: 
                            raise ValueError(f"Not Implemented!")
                        eigenvalue_dict[outer_product] = H_c
        return eigenvalue_dict

    def mean_field_interaction(self, geometry: Geometry, idx_i):
        label_i = geometry.get_label(idx_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        int_parser = eigenvalue_parser.value["interaction"][label_i]
        U_s = int_parser["U_s"]
        U_p = int_parser["U_p"]
        n_s_up, n_s_down = int_parser["n_s_up"], int_parser["n_s_down"]
        n_px_up, n_px_down = int_parser["n_px_up"], int_parser["n_px_down"]
        n_py_up, n_py_down = int_parser["n_py_up"], int_parser["n_py_down"]
        n_pz_up, n_pz_down = int_parser["n_pz_up"], int_parser["n_pz_down"]
        p_interaction = {
            "p_x": (n_px_up, n_px_down),
            "p_y": (n_py_up, n_py_down),
            "p_z": (n_pz_up, n_pz_down),
        }
        E_0 = self.E_0
        eigenvalue_dict = {}
        for sigma in self.spin_dict.values():
            for alpha in self.orbitals:
                outer_product = f"|{alpha}, {sigma}><{alpha}, {sigma}|"
                H_int = 0
                if alpha == "s":
                    n_s = n_s_down if sigma == "+" else n_s_up
                    H_int += U_s * n_s
                elif alpha.startswith('p'):
                    n_p_up, n_p_down = p_interaction[alpha]
                    n_p = n_p_down if sigma == "+" else n_p_up
                    H_int += U_p * n_p
                else: 
                    raise ValueError(f"Not Implemented!")
                eigenvalue_dict[outer_product] = H_int - E_0
        return eigenvalue_dict

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

    def onsite_energy(self, geometry: Geometry, idx_i):
        eigenvalue_dict = {}
        label_i = geometry.get_label(idx_i)
        eigenvalue_parser = getattr(self.cell_parser.eigenvalues, label_i)
        m_parser = eigenvalue_parser.value["onsite_energy"][label_i]
        E_s = m_parser["E_s"]
        E_p = m_parser["E_p"]
        for n, sigma_1 in enumerate(self.spin_dict.values()):
            for alpha in self.orbitals:
                outer_product = f"|{alpha}, {sigma_1}><{alpha}, {sigma_1}|"
                E = 0
                if alpha == "s":
                    E += E_s
                elif alpha.startswith('p'):
                    E += E_p
                else: 
                    raise ValueError(f"Not Implemented!")
                eigenvalue_dict[outer_product] = E
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

    def weight(self, k_idx, site_idx, band):
        if self.band_structure_data is None:
            return None
        N_projections = len(self.coupled_states)
        Psi_dict = self.band_structure_data["eigenvector_dict"]
        Psi_k = Psi_dict[band][k_idx]
        start = site_idx * N_projections
        end = start + N_projections
        c_k = Psi_k[start:end]
        return np.sum(np.abs(c_k)**2)

    @abstractmethod
    def plot_dispersion(self, geometry: Geometry):
        raise NotImplementedError("Implement dispersion plot method!")
