class Parameter:
    def __init__(self, name:str, values:dict):
        self.name = name
        self.setup_values(values)

    def setup_values(self, data_values:dict):
        for parameter, value in data_values.items():
            if isinstance(value["value"], list) and isinstance(value["value"][0], list):
                value["value"] = [
                    [eval(j) if isinstance(j, str) else j for j in i] for i in value["value"]
                ]
            elif isinstance(value["value"], list) and isinstance(value["value"][0], str):
                value["value"] = [eval(i) if isinstance(i, str) else i for i in value["value"]]
            setattr(self, parameter, ParameterAttributes(value))   

class ParameterAttributes:
    def __init__(self, values: dict):
        for key, value in values.items():
            setattr(self, key, value)
