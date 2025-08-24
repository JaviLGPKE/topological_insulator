import numpy as np
import os
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
        self.occupations = np.array(occupations)
        self.N_projections = 8
        self.N_sites = len(occupations)//self.N_projections
        self.location = "bulk"
        self.counter = 0

    # def fitness(self, occupations):
    #     new_occupations = self._objective(occupations)
    #     obj = np.abs(new_occupations - occupations)
        # with open("results/gs_occupations.txt", "a") as f:
        #     f.write(" ".join(map(str, new_occupations)) + "\n")
    #     penalty = 0
    #     for i in range(self.N_sites):
    #         occupation_i = sum(new_occupations[i*self.N_projections:(i+1)*self.N_projections])
    #         if not np.isclose(occupation_i, 1):
    #             penalty += np.inf
    #     fitness = obj + penalty
    #     print(self.counter)
    #     self.counter += 1
    #     return fitness

    def _objective(self, occupations):
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
        occupations_new = self.get_occupations(g, tb_bulk)
        return occupations_new
    
    def get_occupations(self, g, tb_bulk):
        M = tb_bulk.C @ tb_bulk.A
        M_sub = np.kron(np.eye(self.N_sites), M.conj().T)
        N_projections = len(tb_bulk.coupled_states)
        N_sites = len(tb_bulk.sublattice_idxs)
        N_bands = N_sites * N_projections
        N_k = g.N_k
        kx = g.kx_bulk
        ky = g.ky_bulk
        occupations = np.zeros(N_bands)
        for i in range(N_k):
            for j in range(N_k):
                key = f"[{kx[i]}, {ky[j]}]"
                U_k = tb_bulk.U_k_dict[key]
                c = M_sub @ U_k[:, N_bands-1]
                occupations += np.abs(c)**2  # accumulate per orbital & spin
        occupations /= (N_k**2)
        return 

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

