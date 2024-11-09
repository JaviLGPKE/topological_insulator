from acceptor_TI import CellParser, Problem
from IPython import embed

cell_parser = CellParser(f"../data/", "graphene.json")

print(f"{cell_parser.__dir__()}")

print(f"{cell_parser.eigenvalues.__dir__()}")

print(f"{cell_parser.structure.lattice_vectors.value=}")

problem = Problem(cell_parser, save_path=f"results/{cell_parser.general.name.value}")

