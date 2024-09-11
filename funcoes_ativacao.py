import numpy as np

# Função Sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Assumingo que x já passou pela sigmoide
def sigmoid_derivada(x):
    return x * (1 - x)

# Função ReLU e sua derivada
def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return np.where(x > 0, 1, 0)

# Função Linear e sua derivada
def linear(x):
    return x

def linear_derivada(x):
    return np.ones_like(x)

# Função Tanh e sua derivada
def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - np.tanh(x) ** 2
