import scipy.linalg
from topological_insulator import Problem

data_path = "../../../topological_insulator/data/"
file_name = "kagome.json"

problem = Problem(data_path=data_path, file_name=file_name)

location = "edge"
problem.setup(
    N_r = 10,
    N_k = 200, 
    location = location,
    BZ = "reduced"
)

g = problem.geometry
tb = problem.hamiltonian[location]["tight_binding"]

# problem.model_options.solve_connectivity = True

problem.run(
    H_type="reciprocal_space"
)

# idx_i = tb.sublattice_idxs[0]
# label = g.get_label(idx_i)
# # Iterate of bonds with corresponding phases
# site_dict_i = tb.site_data_dict[idx_i]
# dm_dict = site_dict_i["dm_dict"]
# unit_cell_idxs = [idx for idx in dm_dict.keys() if idx in tb.sublattice_idxs]
# non_unit_cell_idxs = [idx for idx in dm_dict.keys() if idx not in tb.sublattice_idxs]
# soi = [idx_i]
# phase_dict = g._get_phase_idxs(idx_i, site_dict_i["dm_dict"], tb.sublattice_idxs)
# for idx_j, idx_j_phase in phase_dict.items():
#     soi.append(idx_j)
#     if idx_j_phase is not None:
#         # soi.append(idx_j)
#         soi.append(idx_j_phase)

# g.plot_lattice(sites_of_interest=soi)