import numpy as np
import os
from scipy.optimize import brentq
from topological_insulator import Problem

class MeanFieldProblem():
    def __init__(
            self, structure_path, structure_name,
            Delta_SOC, t, U, delta, occupations=[],
        ):
        self.structure_path = structure_path
        self.structure_name = structure_name
        self.Delta_SOC = Delta_SOC
        self.t = t
        self.U = U
        self.delta = delta
        self.occupations = np.array(occupations)
        self.N_projections = 8
        self.N_sites = len(occupations)//self.N_projections
        self.location = "bulk"
        self.counter = 0

    # def fitness(self, occupations):
    #     new_occupations = self._objective(occupations)
    #     obj = np.abs(new_occupations - occupations)
    #     with open("results/gs_occupations.txt", "a") as f:
    #         f.write(" ".join(map(str, new_occupations)) + "\n")
    #     penalty = 0
    #     for i in range(self.N_sites):
    #         occupation_i = sum(new_occupations[i*self.N_projections:(i+1)*self.N_projections])
    #         if not np.isclose(occupation_i, 1):
    #             penalty += np.inf
    #     fitness = obj + penalty
    #     print(self.counter)
    #     self.counter += 1
    #     return fitness

    def _objective(self, occupations, E_max, E_min, eta, mu_max, mu_min, T = 300, N_e = 2):
        location = self.location
        print("")
        problem = Problem(
            structure_path=self.structure_path, structure_name=self.structure_name)
        self._set_eigenvalues(problem, occupations)
        problem.setup(
            N_r = 10,
            N_k = 200,
            location = location,
            BZ = "reduced"
        )
        problem.run(
            H_type="reciprocal"
        )
        g = problem.geometry
        tb_bulk = problem.hamiltonian[location]["tight_binding"]
        invariants = problem.hamiltonian["bulk"]["topological_invariants"]
        E, DOS = self.density_of_states(g, tb_bulk, invariants, E_max, E_min, N_E=1000, eta=eta)
        mu = self.find_chemical_potential(E, DOS, N_e, T, mu_max, mu_min)
        occupations_new = self.get_occupations(g, tb_bulk, mu, T)
        return occupations_new, mu
    
    def density_of_states(self, g, tb_bulk, invariants, E_max=12, E_min=-2, N_E=1000, eta=0.08):
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        N_k = g.N_k
        kx = g.kx_bulk
        ky = g.ky_bulk
        E = np.linspace(E_min, E_max, N_E)
        DOS = np.zeros_like(E)
        self.bz_mask = self.brillouin_zone_mask(kx=kx, ky=ky, b1=g.b1, b2=g.b2, M=2)
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not self.bz_mask[ix, iy]:
                    continue
                key =  f"[{k_x}, {k_y}]"
                E_k = tb_bulk.E_k_dict[key]
                for band in range(N_bands):
                    DOS += invariants._lorentz(E, E_k[band], eta)
        DOS /= N_k**2
        return E, DOS
    
    def brillouin_zone_mask(self, kx, ky, b1, b2, M=2, tol=1e-12):
        """
        Wigner-Seitz construction of the first BZ i.e.
        the set of k closer to Gamma(0,0) than any other reciprocal vector.

        returns: boolean mask shape=(len(kx), len(ky)) -> True if k in first BZ.
        """
        N_kx = len(kx); N_ky = len(ky)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        K = np.stack([KX.ravel(), KY.ravel()], axis=1)
        # build small set of reciprocal lattice points R = m*b1 + n*b2
        ms = np.arange(-M, M+1)
        ns = np.arange(-M, M+1)
        R_list = []
        for m in ms:
            for n in ns:
                R_list.append(m * np.asarray(b1) + n * np.asarray(b2))
        R = np.vstack(R_list)  # (numR, 2)
        zero_idx = np.all(np.abs(R) < 1e-12, axis=1)
        displacements = R[~zero_idx] # don't compare with itself
        dist0 = np.sum(K**2, axis=1)
        distR = np.sum((K[:, None, :] - displacements[None, :, :])**2, axis=2)
        # inside BZ iff dist0 <= all distR
        inside_flat = np.all(dist0[:, None] <= distR + tol, axis=1)
        mask = inside_flat.reshape(N_kx, N_ky)
        return mask

    def find_chemical_potential(self, E, DOS, N_e, T=300, mu_max=10, mu_min=5):
        """
        Solve for chemical potential mu such that the integrated number of electrons matches nelecs.

        energies : 1d array (sorted ascending)
        dos      : 1d array, same length, DOS(E)
        nelecs   : target total number of electrons
        T        : temperature in Kelvin
        mu_min, mu_max : search bracket. If None, use energy bounds.
        """
        if mu_min is None:
            mu_min = min(E) - 10.0  # below the lowest energy
        if mu_max is None:
            mu_max = max(E) + 10.0  #  above the highest energy
        objective = lambda mu: self._estimate_N_e(E, DOS, mu, T) - N_e
        mu, result = brentq(objective, mu_min, mu_max, full_output=True)
        if not result.converged:
            raise RuntimeError("Chemical potential solver did not converge.")
        return mu
    
    def _estimate_N_e(self, E, DOS, mu, T):
        return np.trapezoid(DOS * self._fermi_dirac_distribution(E, mu, T), E)

    def _fermi_dirac_distribution(self, E, mu, T):
        k_B = 8.617333262e-5  # eV/K (Boltzmann constant)
        beta = 1.0 / (k_B * T)
        if T <= 0.0:
            return (E <= mu).astype(float)
        else:
            return 1.0 / (np.exp((E - mu)*beta) + 1.0)

    def get_occupations(self, g, tb_bulk, mu, T):
        M = tb_bulk.C @ tb_bulk.A
        M_sub = np.kron(np.eye(self.N_sites), M.conj().T)
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        N_k = g.N_k
        kx = g.kx_bulk
        ky = g.ky_bulk
        occupations = np.zeros(N_bands)
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not self.bz_mask[ix, iy]:
                    continue
                key =  f"[{k_x}, {k_y}]"
                U_k = tb_bulk.U_k_dict[key]
                E_k = tb_bulk.E_k_dict[key]
                for band in range(N_bands):
                    E_k_m = E_k[band]
                    c_k_m = M_sub @ U_k[:, band]
                    occupations += (
                        np.abs(c_k_m)**2 * self._fermi_dirac_distribution(E_k_m, mu, T)
                    )
        occupations /= (N_k**2)
        return occupations

    def _set_eigenvalues(self, problem:Problem, occupations):
        sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        cell = problem.cell_parser
        g = cell.geometry
        n_subs = len(g.delta_vectors.value)
        subs = sublattice_labels[:n_subs]
        for i, label_i in enumerate(subs):
            parser = getattr(problem.cell_parser.eigenvalues, label_i).value
            # Diagonal Values
            parser["chadi_soc"][label_i]["Delta_pp"] = self.Delta_SOC
            parser["interaction"][label_i]["U_p"] = self.U
            parser["interaction"][label_i]["n_px_up"] = occupations[(2)*(i+1)]
            parser["interaction"][label_i]["n_px_down"] = occupations[(3)*(i+1)]
            parser["interaction"][label_i]["n_py_up"] = occupations[(4)*(i+1)]
            parser["interaction"][label_i]["n_py_down"] = occupations[(5)*(i+1)]
            parser["interaction"][label_i]["n_pz_up"] = occupations[(6)*(i+1)]
            parser["interaction"][label_i]["n_pz_down"] = occupations[(7)*(i+1)]
            # Off-Diagonal Values
            for label_j in subs:
                # Hoppings
                try:
                    parser["nn_hopping"][label_j]["t_pp_sigma"] = self.t - self.delta
                    parser["nn_hopping"][label_j]["t_pp_pi"] = self.t + self.delta
                except:
                    pass

    def get_bounds(self):
        n = len(self.occupations)
        return ([0]*n, [1]*n)

    def get_nec(self):
        
        return 0#
    
    def get_nic(self):
        return 0
    
    def get_nobj(self):
        n = len(self.occupations)
        N_sites = n//self.N_projections
        return len(self.occupations)

