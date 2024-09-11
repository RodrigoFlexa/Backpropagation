import numpy as np

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

valores = [7,4.81,0.65]

for i in valores:
    print(f"Valor do {i}", round(sigmoid(i),4))