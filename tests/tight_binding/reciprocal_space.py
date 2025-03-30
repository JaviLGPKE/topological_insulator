import scipy.linalg
from acceptor_TI import Problem

data_path = "../../../acceptor_TI/data/"
file_name = "honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

location = "bulk"
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
    acceptor=False,
    H_type="reciprocal_space"
)