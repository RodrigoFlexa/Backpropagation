import numpy as np

class Normalizador:
    def __init__(self):
        pass
    
    def normaliza(self, data):
        """
        Normaliza os dados para que todos os valores estejam no intervalo [0, 1].
        :param data: np.array com os dados a serem normalizados.
        :return: np.array com os dados normalizados.
        """
        min_val = np.min(data, axis=0) 
        max_val = np.max(data, axis=0) 
        
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        
        # Normaliza os dados
        normalized_data = (data - min_val) / range_val
        return normalized_data
