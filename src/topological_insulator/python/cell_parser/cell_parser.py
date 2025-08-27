import json
import os

from .parameter import Parameter

class CellParser:
    def __init__(self, structure_path, structure_name, material_path, material_name):
        self.sublattice_labels = ["A", "B", "C", "D", "E", "F"]
        self.load_structure(structure_path, structure_name)
        self.eigenvalue_dict = None
        if material_path != "":
            self.eigenvalue_dict = self.load_eigenvalues(material_path, material_name)

    def load_structure(self, structure_path, structure_name):
        path = os.path.join(structure_path, structure_name)
        if os.path.exists(path):
            with open(path, 'r') as file:
                json_data: dict = json.load(file)
        else:
            raise ValueError("Data path does not exist!")

        for hyperparameter, values in json_data.items():
            setattr(self, hyperparameter, Parameter(hyperparameter, values))
    
    def load_eigenvalues(self, material_path, material_name):
        path = os.path.join(material_path, material_name)
        if os.path.exists(path):
            with open(path, 'r') as file:
                json_data: dict = json.load(file)
        else:
            raise ValueError("Data path does not exist!")
        return json_data

    def set_eigenvalues(self):
        g = self.geometry
        g.lattice_constant.value = self.eigenvalue_dict["lattice_constant"]
        g.buckling_cosine.value = self.eigenvalue_dict["buckling_cosine"]
        eigenvalue_dict = self.eigenvalue_dict["eigenvalues"]
        sublattice_labels = self.sublattice_labels
        n_subs = len(g.delta_vectors.value)
        subs = sublattice_labels[:n_subs]
        for label_i in subs:
            parser = getattr(self.eigenvalues, label_i).value
            # Diagonal Values
            parser["onsite_energy"][label_i]["E_s"] = eigenvalue_dict["E_s"]
            parser["onsite_energy"][label_i]["E_p"] = eigenvalue_dict["E_p"]
            parser["chadi_soc"][label_i]["Delta_pp"] = eigenvalue_dict["Delta_pp"]
            # Off-Diagonal Values
            for label_j in subs:
                # Hoppings
                try:
                    parser["nn_hopping"][label_j]["t_ss_sigma"] = eigenvalue_dict["t_ss_sigma"]
                    parser["nn_hopping"][label_j]["t_sp_sigma"] = eigenvalue_dict["t_sp_sigma"]
                    parser["nn_hopping"][label_j]["t_pp_sigma"] = eigenvalue_dict["t_pp_sigma"]
                    parser["nn_hopping"][label_j]["t_pp_pi"] = eigenvalue_dict["t_pp_pi"]
                except:
                    pass
                try:
                    parser["kane_mele_soc"][label_i]["lambda_ss"] = eigenvalue_dict["lambda_ss"]
                    parser["kane_mele_soc"][label_i]["lambda_pp"] = eigenvalue_dict["lambda_pp"]
                except:
                    pass
        

    
