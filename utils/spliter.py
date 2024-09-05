import pandas as pd
import numpy as np

class DataSplitter:
    def __init__(self, dataset):
        """
        Inicializa a classe com o dataset. O dataset deve ser um DataFrame.
        """
        self.dataset = dataset.values  # Converte o DataFrame para NumPy
        self.columns = dataset.columns  # Armazena os nomes das colunas do DataFrame original
    
    def split_data(self, train_pct, test_pct, val_pct):
        """
        Divide os dados em treino, teste e validação com base nas porcentagens fornecidas.
        
        :param train_pct: Porcentagem de amostras para treino (entre 0 e 1)
        :param test_pct: Porcentagem de amostras para teste (entre 0 e 1)
        :param val_pct: Porcentagem de amostras para validação (entre 0 e 1)
        :return: DataFrames para treino, teste e validação
        """
        # Verificar se as porcentagens somam 1
        if not np.isclose(train_pct + test_pct + val_pct, 1.0):
            raise ValueError("As porcentagens devem somar 1.")
        
        # Calcular o número de amostras para cada conjunto
        n_total = self.dataset.shape[0]
        n_train = int(train_pct * n_total)
        n_test = int(test_pct * n_total)
        n_val = n_total - n_train - n_test  # O restante será para validação
        
        # Embaralhar os dados
        np.random.shuffle(self.dataset)
        
        # Dividir os dados
        train_data = self.dataset[:n_train]
        test_data = self.dataset[n_train:n_train + n_test]
        val_data = self.dataset[n_train + n_test:]
        
        # Reconverter para DataFrames
        train_df = pd.DataFrame(train_data, columns=self.columns)
        test_df = pd.DataFrame(test_data, columns=self.columns)
        val_df = pd.DataFrame(val_data, columns=self.columns)
        
        return train_df, test_df, val_df
