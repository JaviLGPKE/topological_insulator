from acceptor_TI import Problem

data_path = "../../../acceptor_TI/data/"
file_name = "honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

problem.setup(
    size = 10,
    N_k = 300, 
    dispersion = True,
)

problem.run()

problem.plot(type="dispersion")