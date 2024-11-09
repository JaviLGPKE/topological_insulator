import numpy as np
from matplotlib import pyplot as plt

# import kwant
# import quspin

from .cell_parser import CellParser

class Problem:
    def __init__(self, cell_parser:CellParser, save_path=None):
        self.cell_parser = cell_parser
        # Matrices
        self.I = np.identity(self.cell_parser.general.dimensions.value)
        self.s_x = np.array([[0, 1], [1, 0]])
        self.s_y = np.array([[0, -1j], [1j, 0]])
        self.s_z = np.diag([1, -1])

    # def next_nearest_neighbours(self):
    #     A, B = self.lattice.sublattices 
    #     NN_A = (((-1, 0), A, A), ((0, 1), A, A), ((1, -1), A, A))
    #     NN_B = (((1, 0), B, B), ((0, -1), B, B), ((-1, 1), B, B))
    #     return NN_A + NN_B
    
    # def on_site(self, site, parameters):
    #     A, B = self.lattice.sublattices 
    #     return parameters.m * (1 if site.family == A else -1)
        