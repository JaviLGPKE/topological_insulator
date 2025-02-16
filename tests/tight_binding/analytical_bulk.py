from acceptor_TI import Problem

data_path = "../../../acceptor_TI/data/"
file_name = "honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

hopping_dict = problem.cell_parser.eigenvalues.nn_hopping.value

# hopping_dict["t_ss_sigma"] = -1.4
hopping_dict["t_sp_sigma"] = 1
# hopping_dict["t_pp_sigma"] = 2
# hopping_dict["t_pp_pi"] = -3

problem.setup(
    size = 10,
    N_k = 300, 
    dispersion = True,
)


problem.run(
    acceptor=False,
    type="bulk"
)

# problem.plot(type="analytical_dispersion")