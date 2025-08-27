import numpy as np
import os
from scipy.optimize import brentq, minimize_scalar
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

    def setup(self, E_max, E_min, eta, T = 300, N_h = 2):
        self.E_max = E_max
        self.E_min = E_min
        self.eta = eta
        self.T = T
        self.N_h = N_h

    def fitness(self, occupations):
        location = self.location
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
        E, DOS = self.density_of_states(
            g, tb_bulk, invariants, self.E_max, self.E_min, N_E=1000, eta=self.eta)
        mu_min = np.min(E) - 10
        mu_max = np.max(E) + 10
        mu = self.find_chemical_potential(E, DOS, self.N_h, self.T, mu_max, mu_min)
        occ_e, occ_h = self.get_occupations(g, tb_bulk, E, mu, self.T)
        F = self.helmholtz_free_energy(g, tb_bulk, E, mu, self.T)
        diff = np.abs(np.sum(occ_h)-self.N_h)
        return [F, diff]
    
    def density_of_states(self, g, tb_bulk, invariants, E_max=12, E_min=-2, N_E=1000, eta=0.08):
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        kx = g.kx_bulk
        ky = g.ky_bulk
        E = np.linspace(E_min, E_max, N_E)
        DOS = np.zeros_like(E)
        N = 0
        for ix, k_x in enumerate(kx):
            for iy, k_y in enumerate(ky):
                if not g.BZ_mask[ix, iy]:
                    continue
                key =  f"[{k_x}, {k_y}]"
                for band in range(N_bands):
                    E_k_m = tb_bulk.E_k_dict[key][band]
                    if E_k_m > E_max or E_k_m < E_min:
                        continue
                    DOS += invariants._lorentz(E, E_k_m, eta)
                N += 1        
        DOS /= N
        return E, DOS

    def find_chemical_potential(self, E, DOS, N_h, T=300, mu_max=10, mu_min=5):
        """
        Solve for chemical potential mu such that the integrated number of electrons matches N_h.
        """
        objective = lambda mu: self._estimate_N_h(E, DOS, mu, T) - N_h
        # res = minimize_scalar(objective, bounds=(mu_min, mu_max), method='bounded',
        #                   options={'xatol':1e-6})
        # if not res.success:
        #     raise RuntimeError("minimize_scalar failed: " + getattr(res, "message", "no message"))
        # mu = float(res.x)
        # try:
        #     a = max(mu_min, mu - 1.0)
        #     b = min(mu_max, mu + 1.0)
        #     fa = self._estimate_N_h(E, DOS, a, T) - N_h
        #     fb = self._estimate_N_h(E, DOS, b, T) - N_h
        #     if fa * fb <= 0:
        #         mu = float(brentq(lambda m: self._estimate_N_h(E, DOS, m, T) - N_h, a, b))
        # except Exception:
        #     pass
        mu, result = brentq(objective, mu_min, mu_max, full_output=True)
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
        N = 0
        occ_e, occ_h = np.zeros(N_bands), np.zeros(N_bands)
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
                    f_E = self._fermi_dirac_distribution(E_k_m, mu, T)
                    c_k_m = M_sub @ U_k[:, band]
                    weight = np.abs(c_k_m)**2 
                    occ_e += weight * f_E
                    occ_h += weight * (1-f_E)
                N +=1
        occ_e /= N
        occ_h /= N
        return occ_e, occ_h

    def helmholtz_free_energy(self, g, tb_bulk, E, mu, T):
        E_max, E_min = max(E), min(E)
        kx = g.kx_bulk
        ky = g.ky_bulk
        N_bands = len(next(iter(tb_bulk.E_k_dict.values())))
        E_band_sum = 0.0
        S_sum = 0.0
        N = 0
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
                    # if not np.isclose(T, 0, rtol=1e-3):
                        # S_sum += - (f_E * np.log(f_E) + (1 - f_E) * np.log(1 - f_E))
                N += 1
        E_0 = E_band_sum / N
        #S = self.k_B * (S_sum / N)
        return E_0 #- (T * S)

    def _set_eigenvalues(self, problem:Problem, occupations, debug:bool=False):
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
            if debug:
                print(parser)

    def get_bounds(self):
        n = len(self.occupations_ini)
        return ([0]*n, [0.8]*n)

    def get_nec(self):
        return 1
    
    def get_nic(self):
        return 0
    
    def get_nobj(self):
        return 1