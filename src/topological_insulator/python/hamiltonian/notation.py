import numpy as np
import re
from matplotlib import pyplot as plt

class Notation:

    def __init__(self):
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        self.direction_index = {'x': 0, 'y': 1, 'z': 2}
        # Spin
        self.n_spins = 2
        self.spin_dict = {1/2: "+", -1/2: "-"}
        self.pauli_matrix_dict = {
            0: np.array(
                [[0, 1],
                [1, 0]]),
            1: np.array(
                [[0, -1j],
                 [1j, 0]]),
            2: np.array(
                [[1, 0],
                [0, -1]])
        }
        self.angular_momentum_operator_dict = {
            0: np.array(
                [[0, 0, 0],
                [0, 0,-1j],
                [0, 1j, 0]]),
            1: np.array(
                [[0, 0, 1j],
                [0, 0, 0 ],
                [-1j, 0, 0]]),
            2: np.array(
                [[0, -1j, 0],
                [1j, 0, 0],
                [0, 0, 0]])
        }
        # Orbitals
        self.orbitals = ['s', 'p_x', 'p_y', 'p_z']
        self.n_orbitals = len(self.orbitals)
        self.state_pattern = re.compile(r'\|([\d\.\-]+),([\d\.\-]+);([\d\.\-]+),([\d\.\-]+)\>')

    def l_to_orbitals(self, l, m_l):
        """
        Convert an angular momentum state |l, m_l> into a linear combination
        of orbital states. Returns a dictionary where the keys are the orbital
        labels ('s', 'p_x', 'p_y', 'p_z') and the values are the expansion coefficients.
        
        Using the conventions:
        |0,0>         = |s>
        |1,0>         = |p_z>
        |1,+1>        = -1/sqrt(2) ( |p_x> + |p_y> )
        |1,-1>        = +1/sqrt(2) ( |p_x> - |p_y> )
        """
        # s
        if l == 0 and m_l == 0:
            return {"s": 1.0}
        # p
        elif l == 1:
            if m_l == 0:
                return {"p_z": 1.0}
            elif m_l == 1:
                return {"p_x": -1/np.sqrt(2), "p_y": 1j* -1/np.sqrt(2)}
            elif m_l == -1:
                return {"p_x":  1/np.sqrt(2), "p_y": 1j* -1/np.sqrt(2)}
            else:
                raise ValueError("Invalid m_l value for l=1")
        else:
            raise ValueError("Conversion for l > 1 is not implemented!")      

    def get_quantum_number(self, key, pos=0):
        """
        Extract a quantum number from a state string in ket notation: |a,b;c,d>
        The 'pos' parameter determines which quantum number to extract:
            pos = 0: returns the first quantum number (a)
            pos = 1: returns the second quantum number (b)
            pos = 2: returns the third quantum number (c)
            pos = 3: returns the fourth quantum number (d)
        
        Parameters:
            key (str): The state string.
            pos (int): The 0-indexed position of the quantum number to extract.
            
        Returns:
            float: The quantum number at the specified position.
            
        Raises:
            ValueError: If the state string format is incorrect or if 'pos' is out of range.
        """
        match = self.state_pattern.search(key)
        if match:
            try:
                # Adjust for 1-indexed regex groups
                return float(match.group(pos + 1))
            except IndexError:
                raise ValueError(f"State string does not contain a quantum number at position {pos}")
        else:
            raise ValueError("State string format is incorrect.")
    
    def _visualise_matrix(self, M):
        plt.figure(figsize=(12, 5))
        cmap = plt.get_cmap('coolwarm')
        cmap.set_bad('white')
        mask = np.isclose(M, 0, atol=1e-12)
        # M_masked = np.ma.masked_where(mask, M)
        plt.subplot(1, 2, 1)
        plt.imshow(M.real, cmap=cmap)
        plt.title('Real Part')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(M.imag, cmap=cmap)
        plt.title('Imaginary Part')
        plt.colorbar()
        plt.show()