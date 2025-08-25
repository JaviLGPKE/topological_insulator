import numpy as np
import os
from scipy.optimize import brentq, minimize, basinhopping
from topological_insulator import Problem

class MeanFieldProblem():
    def __init__(
            self, structure_path, structure_name,
            Delta_SOC, t, U, delta, occupations=[]
        ):
        self.structure_path = structure_path
        self.structure_name = structure_name
        self.Delta_SOC = Delta_SOC
        self.t = t
        self.U = U
        self.delta = delta
        self.occupations_ini = np.array(occupations)
        self.N_projections = 8
        self.N_sites = len(occupations)//self.N_projections
        self.location = "bulk"
        self.counter = 0
        self.k_B = 8.617333262e-5  # eV/K (Boltzmann constant)

    def run(self, N_h=2, T=300, E_max=10, E_min=1, eta= 0.08, mu_max=10, mu_min=1):
        N_occ = len(self.occupations_ini)
        lower, upper = 0, 1
        bounds = [(lower, upper)] * N_occ
        x0 = np.zeros(N_occ)
        x0 += 1e-3 * np.random.randn(N_occ)
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options": {"ftol": 1e-3}}
        print("Global Optimization...")
        bh = basinhopping(
            lambda x: self._objective(x, E_max, E_min, eta, mu_max, mu_min, T, N_h),
            x0, niter=20, minimizer_kwargs=minimizer_kwargs, stepsize=0.05) # global
        print("Global Optimization - Done")
        self.counter = 0
        print("Local Optimization...")
        res_local = minimize(
            lambda x: self._objective(x, E_max, E_min, eta, mu_max, mu_min, T, N_h),
            bh.x, method="L-BFGS-B", bounds=bounds, 
            options={"maxiter": 100, "ftol": 1e-4})  # local
        print("Local Optimization - Done")
        occupations = res_local.x
        F = res_local.fun
        return occupations, F

    def _objective(self, occupations, E_max, E_min, eta, mu_max, mu_min, T = 300, N_h = 2):
        print(f"Iteration: {self.counter}")
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
        mu_min = np.min(E) 
        mu_max = np.max(E)
        mu = self.find_chemical_potential(E, DOS, N_h, T, mu_max, mu_min)
        # occupations_new = self.get_occupations(g, tb_bulk, E, mu, T)
        self.counter += 1
        return self.helmholtz_free_energy(g, tb_bulk, E, mu, T)
    
    def density_of_states(self, g, tb_bulk, invariants, E_max=12, E_min=-2, N_E=1000, eta=0.08):
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        kx = g.kx_bulk
        ky = g.ky_bulk
        E = np.linspace(E_min, E_max, N_E)
        DOS = np.zeros_like(E)
        self.N_k_BZ = 0
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not g.BZ_mask[ix, iy]:
                    continue
                self.N_k_BZ += 1
                key =  f"[{k_x}, {k_y}]"
                E_k = tb_bulk.E_k_dict[key]
                for band in range(N_bands):
                    DOS += invariants._lorentz(E, E_k[band], eta)
        DOS /= self.N_k_BZ
        return E, DOS

    def find_chemical_potential(self, E, DOS, N_h, T=300, mu_max=10, mu_min=5):
        """
        Solve for chemical potential mu such that the integrated number of electrons matches nelecs.

        energies : 1d array (sorted ascending)
        dos      : 1d array, same length, DOS(E)
        nelecs   : target total number of electrons
        T        : temperature in Kelvin
        mu_min, mu_max : search bracket. If None, use energy bounds.
        """
        objective = lambda mu: self._estimate_N_h(E, DOS, mu, T) - N_h
        mu, result = brentq(objective, mu_min, mu_max, full_output=True)
        if not result.converged:
            raise RuntimeError("Chemical potential solver did not converge.")
        return mu
    
    def _estimate_N_h(self, E, DOS, mu, T):
        y = DOS * (1-self._fermi_dirac_distribution(E, mu, T))
        x = E
        return np.trapezoid(y, x)

    def _fermi_dirac_distribution(self, E, mu, T):
        beta = 1.0 / (self.k_B * T)
        if T <= 0.0:
            return (E <= mu).astype(float)
        else:
            return 1.0 / (np.exp((E - mu)*beta) + 1.0)

    def get_occupations(self, g, tb_bulk, E, mu, T):
        E_max, E_min = max(E), min(E)
        M = tb_bulk.C @ tb_bulk.A
        M_sub = np.kron(np.eye(self.N_sites), M.conj().T)
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        kx = g.kx_bulk
        ky = g.ky_bulk
        occupations = np.zeros(N_bands)
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not g.BZ_mask[ix, iy]:
                    continue
                key =  f"[{k_x}, {k_y}]"
                U_k = tb_bulk.U_k_dict[key]
                E_k = tb_bulk.E_k_dict[key]
                for band in range(N_bands):
                    E_k_m = E_k[band]
                    if E_k_m > E_max or E_k_m < E_min:
                        continue
                    c_k_m = M_sub @ U_k[:, band]
                    occupations += (
                        np.abs(c_k_m)**2 * self._fermi_dirac_distribution(E_k_m, mu, T)
                    )
        occupations /= self.N_k_BZ
        return occupations
    
    def helmholtz_free_energy(self, g, tb_bulk, E, mu, T):
        E_max, E_min = max(E), min(E)
        kx = g.kx_bulk
        ky = g.ky_bulk
        N_bands = len(next(iter(tb_bulk.E_k_dict.values())))
        E_band_sum = 0.0
        S_sum = 0.0
        Nk = 0
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not g.BZ_mask[ix, iy]:
                    continue
                key = f"[{k_x}, {k_y}]"
                E_k = tb_bulk.E_k_dict[key]
                for band in range(N_bands):
                    E_k_m = E_k[band]
                    if E_k_m > E_max or E_k_m < E_min:
                        continue
                    f_E = self._fermi_dirac_distribution(E_k_m, mu, T)
                    E_band_sum += E_k_m * f_E
                    if T !=0:
                        S_sum += - (f_E * np.log(f_E) + (1 - f_E) * np.log(1 - f_E))
        E_band = E_band_sum / self.N_k_BZ
        S = self.k_B * (S_sum / self.N_k_BZ)
        return E_band - (T * S)

    def _set_eigenvalues(self, problem:Problem, occupations):
        sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        cell = problem.cell_parser
        g = cell.geometry
        n_subs = len(g.delta_vectors.value)
        subs = sublattice_labels[:n_subs]
        for i, label_i in enumerate(subs):
            parser = getattr(problem.cell_parser.eigenvalues, label_i).value
            # Diagonal Values
            base = i * self.N_projections 
            parser["chadi_soc"][label_i]["Delta_pp"] = self.Delta_SOC
            parser["interaction"][label_i]["U_p"] = self.U
            parser["interaction"][label_i]["n_px_up"] = occupations[2+base]
            parser["interaction"][label_i]["n_px_down"] = occupations[3+base]
            parser["interaction"][label_i]["n_py_up"] = occupations[4+base]
            parser["interaction"][label_i]["n_py_down"] = occupations[5+base]
            parser["interaction"][label_i]["n_pz_up"] = occupations[6+base]
            parser["interaction"][label_i]["n_pz_down"] = occupations[7+base]
            # Off-Diagonal Values
            for label_j in subs:
                # Hoppings
                try:
                    parser["nn_hopping"][label_j]["t_pp_sigma"] = self.t - self.delta
                    parser["nn_hopping"][label_j]["t_pp_pi"] = self.t + self.delta
                except:
                    pass

    def get_bounds(self):
        n = len(self.occupations_ini)
        return ([0]*n, [1]*n)

    def get_nec(self):
        
        return 0#
    
    def get_nic(self):
        return 0
    
    def get_nobj(self):
        return 1

