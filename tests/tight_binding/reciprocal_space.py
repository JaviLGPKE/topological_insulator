from acceptor_TI import Problem

data_path = "../../../acceptor_TI/data/"
file_name = "honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

hopping_dict = problem.cell_parser.eigenvalues.nn_hopping.value
hopping_dict["t_ss_sigma"] = -1.4
hopping_dict["t_sp_sigma"] = 1
hopping_dict["t_pp_sigma"] = 1
hopping_dict["t_pp_pi"] = -0.5

problem.setup(
    N_r = 10,
    N_k = 200, 
    location = "edge"
)

problem.run(
    acceptor=False,
    H_type="reciprocal_space"
)

problem.plot(plot_type="dispersion", location="edge")

# [np.int64(98), np.int64(96), np.int64(94), np.int64(92), 
#  np.int64(90), np.int64(88), np.int64(86), np.int64(84), 
#  np.int64(82), np.int64(80), np.int64(79), np.int64(77), 
#  np.int64(75), np.int64(73), np.int64(71), np.int64(69), 
#  np.int64(67), np.int64(65), np.int64(63), np.int64(61)]

# 98, 79, 99 (0), 96, 77, 97 (0), 75, 94, 95, 73, 92, 93, 90, 71, 88, 69, 89, 86, 67, 