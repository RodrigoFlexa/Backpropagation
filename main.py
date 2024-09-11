import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nn import NeuralNetwork
from utils.normalizador import Normalizador
from utils.spliter import Splitter
from sklearn.metrics import confusion_matrix

SEED = 42
np.random.seed(SEED)

# Ler o dataset e definir os nomes das colunas
dataset = pd.read_excel("dadosmamografia.xlsx")
colunas = ['x1', 'x2', 'x3', 'x4', 'x5', 'y']
dataset.columns = colunas

# Criar uma instância da classe Normalizador
normalizador = Normalizador()

# Normalizar os dados
dataset_normalizado = normalizador.normaliza(dataset.values)
dataset_normalizado = pd.DataFrame(dataset_normalizado, columns=colunas)

# Separar o dataset (60% treino, 20% teste, 20% validação)
splitter = Splitter(dataset_normalizado)
treino, teste, validacao = splitter.split_data(0.6, 0.2, 0.2)

# Exibir o número de amostras em cada conjunto
print(f"Dataset: {dataset_normalizado.shape}")
print(f"Treino: {treino.shape}")
print(f"Teste: {teste.shape}")
print(f"Validação: {validacao.shape}")

# Separar as entradas (X) e saídas (y) do conjunto de treino, teste e validação
X_treino = treino[['x1', 'x2', 'x3', 'x4', 'x5']].values
y_treino = treino[['y']].values

X_validacao = validacao[['x1', 'x2', 'x3', 'x4', 'x5']].values
y_validacao = validacao[['y']].values

X_teste = teste[['x1', 'x2', 'x3', 'x4', 'x5']].values
y_teste = teste[['y']].values

# Criar modelo com 2 camadas escondidas (4 neurônios na primeira, 3 na segunda)
modelo = NeuralNetwork(n_entradas=5, n_saidas=1, n_neuronios_escondidos=16, func_ativacao='relu',seed=SEED)

print()
modelo.mostrar_pesos('inicio')

# Treinar o modelo (com validação cruzada e parada antecipada)
modelo.treinar(X_treino, y_treino, X_validacao, y_validacao, epochs=1000, taxa_aprendizagem=0.01,verbose='n')

print()
modelo.mostrar_pesos('Fim')

# Fazer previsões no conjunto de teste
previsoes_teste = modelo.prever(X_teste)

# Transformar as previsões em binárias (0 ou 1)
previsoes_binarias = np.where(previsoes_teste >= 0.5, 1, 0)

# Calcular a acurácia no conjunto de teste
acuracia = np.mean(previsoes_binarias == y_teste)
print(f"Acurácia no conjunto de teste: {acuracia:.6f}")

# # Gerar a matriz de confusão
# matriz_confusao = confusion_matrix(y_teste, previsoes_binarias)

# # Plotar a matriz de confusão
# # Gerar a matriz de confusão
matriz_confusao = confusion_matrix(y_teste, previsoes_binarias)
print(matriz_confusao)

# Plotar os resultados (erro médio quadrático e acurácia)
modelo.plotar_resultados(fontsize=25)
