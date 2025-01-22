from acceptor_TI import Problem

data_path="../../../acceptor_TI/data/"
file_name="honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

problem.geometry.plot_lattice(list_indexes=False)