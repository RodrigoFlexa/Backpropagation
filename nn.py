import numpy as np
import matplotlib.pyplot as plt

# Funções de ativação
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return np.where(x > 0, 1, 0)

class NeuralNetwork:
    def __init__(self, n_entradas, n_saidas, camadas_escondidas, func_ativacao='sigmoid'):
        # Inicialização da rede neural
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.camadas_escondidas = camadas_escondidas
        
        # Configurar função de ativação
        if func_ativacao == 'sigmoid':
            self.func_ativacao = sigmoid
            self.func_ativacao_derivada = sigmoid_derivada
        elif func_ativacao == 'relu':
            self.func_ativacao = relu
            self.func_ativacao_derivada = relu_derivada
        else:
            raise ValueError("Função de ativação não reconhecida: use 'sigmoid' ou 'relu'.")

        # Inicializar pesos e biases para cada camada
        self.pesos = []
        self.biases = []

        # Pesos entre entrada e primeira camada escondida
        self.pesos.append(np.random.randn(n_entradas, camadas_escondidas[0]))
        self.biases.append(np.zeros((1, camadas_escondidas[0])))

        # Pesos e biases entre as camadas escondidas
        for i in range(len(camadas_escondidas) - 1):
            self.pesos.append(np.random.randn(camadas_escondidas[i], camadas_escondidas[i+1]))
            self.biases.append(np.zeros((1, camadas_escondidas[i+1])))

        # Pesos entre última camada escondida e camada de saída
        self.pesos.append(np.random.randn(camadas_escondidas[-1], n_saidas))
        self.biases.append(np.zeros((1, n_saidas)))

        # Armazenar o histórico de erro e acurácia
        self.historico_mse_treino = []
        self.historico_mse_validacao = []
        self.historico_acuracia_validacao = []

    def forward(self, X):
        # Realiza o forward pass
        self.ativacoes = [X]
        self.zs = []

        # Propagar pelas camadas
        entrada = X
        for i in range(len(self.pesos) - 1):
            z = np.dot(entrada, self.pesos[i]) + self.biases[i]
            self.zs.append(z)
            ativacao = self.func_ativacao(z)
            self.ativacoes.append(ativacao)
            entrada = ativacao

        # Camada de saída
        z_saida = np.dot(entrada, self.pesos[-1]) + self.biases[-1]
        self.zs.append(z_saida)
        saida = sigmoid(z_saida)  # Supondo sigmoid para a camada de saída
        self.ativacoes.append(saida)

        return saida

    def backpropagation(self, X, y, taxa_aprendizagem):
        # Realiza o backpropagation para ajustar os pesos e biases
        saida = self.forward(X)
        erro = y - saida

        deltas = [erro * sigmoid_derivada(saida)]

        for i in reversed(range(len(self.pesos) - 1)):
            delta = np.dot(deltas[0], self.pesos[i+1].T) * self.func_ativacao_derivada(self.ativacoes[i+1])
            deltas.insert(0, delta)

        for i in range(len(self.pesos)):
            self.pesos[i] += np.dot(self.ativacoes[i].T, deltas[i]) * taxa_aprendizagem
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * taxa_aprendizagem

    def calcular_acuracia(self, X, y):
        previsoes = self.prever(X)
        previsoes_binarias = np.where(previsoes >= 0.5, 1, 0)
        return np.mean(previsoes_binarias == y)

    def calcular_mse(self, X, y):
        saida = self.forward(X)
        mse = np.mean(np.square(y - saida))
        return mse

    def treinar(self, X, y, X_validacao, y_validacao, epochs=1000, taxa_aprendizagem=0.01, early_stopping_limit=5):
        n_samples = X.shape[0]
        melhor_erro_validacao = float('inf')
        epochs_erro_melhorou = 0

        for epoch in range(epochs):
            for i in range(n_samples):
                xi = X[i].reshape(1, -1)
                yi = y[i].reshape(1, -1)
                self.backpropagation(xi, yi, taxa_aprendizagem)

            # Calcular o erro nos conjuntos de treino e validação
            mse_treino = self.calcular_mse(X, y)
            mse_validacao = self.calcular_mse(X_validacao, y_validacao)
            acuracia_validacao = self.calcular_acuracia(X_validacao, y_validacao)

            # Armazenar os MSEs e a acurácia
            self.historico_mse_treino.append(mse_treino)
            self.historico_mse_validacao.append(mse_validacao)
            self.historico_acuracia_validacao.append(acuracia_validacao)

            print(f"Época {epoch+1}/{epochs}, Erro de Treino MSE: {mse_treino:.6f}, Erro de Validação MSE: {mse_validacao:.6f}, Acurácia de Validação: {acuracia_validacao:.6f}")

            # Parada antecipada
            if mse_validacao < melhor_erro_validacao:
                melhor_erro_validacao = mse_validacao
                epochs_erro_melhorou = 0
            else:
                epochs_erro_melhorou += 1

            if epochs_erro_melhorou >= early_stopping_limit:
                print(f"Parada antecipada na época {epoch+1} devido ao aumento do erro na validação.")
                break

    def prever(self, X):
        return self.forward(X)

    def plotar_resultados(self):
        # Gráfico de MSE (treino e validação)
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_mse_treino, label='Erro de Treino (MSE)', color='blue')
        plt.plot(self.historico_mse_validacao, label='Erro de Validação (MSE)', color='orange')
        plt.title('Evolução do Erro Médio Quadrático (MSE)')
        plt.xlabel('Épocas')
        plt.ylabel('Erro Médio Quadrático')
        plt.legend()
        plt.show()

        # Gráfico de Acurácia
        plt.figure(figsize=(10, 6))
        plt.plot(self.historico_acuracia_validacao, label='Acurácia de Validação', color='green')
        plt.title('Evolução da Acurácia de Validação')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.show()
