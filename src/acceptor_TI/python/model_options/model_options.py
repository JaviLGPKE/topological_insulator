class ModelOptions:
    def __init__(self, size, N_k, location, BZ):
        self.N_r = size
        self.N_k = N_k
        self.location = location
        self.BZ = BZ
        # Admin Debug
        self.solve_connectivity = False
    