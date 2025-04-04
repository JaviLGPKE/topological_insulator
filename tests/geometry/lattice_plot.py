from topological_insulator import Problem

data_path="../../../topological_insulator/data/"
file_name="honeycomb.json"

problem = Problem(data_path=data_path, file_name=file_name)

problem.setup()

problem.plot(type="lattice")