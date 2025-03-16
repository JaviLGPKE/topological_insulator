from acceptor_TI import Problem

data_path = "../../../acceptor_TI/data/"
file_name = "kagome.json"

problem = Problem(data_path=data_path, file_name=file_name)

hopping_dict = problem.cell_parser.eigenvalues.nn_hopping.value
# hopping_dict["t_ss_sigma"] = -1.4
# hopping_dict["t_sp_sigma"] = 1
# hopping_dict["t_pp_sigma"] = 1
# hopping_dict["t_pp_pi"] = -0.5

location = "bulk"
problem.setup(
    N_r = 10,
    N_k = 100, 
    location = location
)

problem.model_options.solve_connectivity = True

problem.run(
    acceptor=False,
    H_type="reciprocal_space"
)

problem.plot(plot_type="dispersion", location=location)