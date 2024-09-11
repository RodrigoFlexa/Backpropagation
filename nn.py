import numpy as np
import matplotlib.pyplot as plt
from funcoes_ativacao import sigmoid, sigmoid_derivada, relu, relu_derivada, linear, linear_derivada, tanh, tanh_derivada

class NeuralNetwork:
    def __init__(self, n_entradas, n_saidas, n_neuronios_escondidos, func_ativacao='sigmoid', seed=None):
        # Inicialização da rede neural com apenas uma camada escondida
        self.n_entradas = n_entradas
        self.n_saidas = n_saidas
        self.n_neuronios_escondidos = n_neuronios_escondidos
        
        if seed is not None:
            np.random.seed(seed)

        # Configurar função de ativação
        if func_ativacao == 'sigmoid':
            self.func_ativacao = sigmoid
            self.func_ativacao_derivada = sigmoid_derivada
        elif func_ativacao == 'relu':
            self.func_ativacao = relu
            self.func_ativacao_derivada = relu_derivada
        elif func_ativacao == 'linear':
            self.func_ativacao = linear
            self.func_ativacao_derivada = linear_derivada
        elif func_ativacao == 'tanh':
            self.func_ativacao = tanh
            self.func_ativacao_derivada = tanh_derivada
        else:
            raise ValueError("Função de ativação não reconhecida: use 'sigmoid', 'relu', 'linear' ou 'tanh'.")

        # Inicializar pesos e biases
        self.pesos = [
            np.random.randn(n_entradas, n_neuronios_escondidos),  # Pesos da entrada para a camada escondida
            np.random.randn(n_neuronios_escondidos, n_saidas)  # Pesos da camada escondida para a saída
        ]
        self.biases = [
            np.zeros((1, n_neuronios_escondidos)),  # Biases da camada escondida
            np.zeros((1, n_saidas))  # Biases da camada de saída
        ]

        # Armazenar o histórico de erro e acurácia
        self.historico_mse_treino = []
        self.historico_mse_validacao = []
        self.historico_acuracia_validacao = []

    def forward(self, X):
        # Realiza o forward pass com apenas uma camada escondida
        self.net_entrada = X
        self.net_oculta = np.dot(X, self.pesos[0]) + self.biases[0]  # Entrada para a camada escondida
        self.out_oculta = self.func_ativacao(self.net_oculta)  # Ativação da camada escondida

        self.net_saida = np.dot(self.out_oculta, self.pesos[1]) + self.biases[1]  # Entrada para a camada de saída

        self.out_saida = sigmoid(self.net_saida)  # Ativação da camada de saída (sigmoid)
        return self.out_saida

    def backpropagation(self, X, y, taxa_aprendizagem):
        saida = self.forward(X)  # Saída da rede
        erro = y - saida  # Erro/derivada da função de custo MSE em relação à saída da rede

        # Passo 1: Calcula o delta da camada de saída
        delta_saida = erro * self.func_ativacao_derivada(saida)

        # Passo 2: Calcula o delta da camada escondida
        delta_oculta = np.dot(delta_saida, self.pesos[1].T) * self.func_ativacao_derivada(self.out_oculta)

        # Passo 3: Atualiza os pesos e biases
        self.pesos[1] += np.dot(self.out_oculta.T, delta_saida) * taxa_aprendizagem
        self.biases[1] += np.sum(delta_saida, axis=0, keepdims=True) * taxa_aprendizagem
        
        self.pesos[0] += np.dot(X.T, delta_oculta) * taxa_aprendizagem
        self.biases[0] += np.sum(delta_oculta, axis=0, keepdims=True) * taxa_aprendizagem

    def calcular_acuracia(self, X, y):
        previsoes = self.prever(X)
        previsoes_binarias = np.where(previsoes >= 0.5, 1, 0)
        return np.mean(previsoes_binarias == y)

    def calcular_mse(self, X, y):
        saida = self.forward(X)
        mse = np.mean(np.square(y - saida))
        return mse

    def treinar(self, X, y, X_validacao, y_validacao, epochs=1000, taxa_aprendizagem=0.01, early_stopping_limit=5, verbose='n'):
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

            if verbose == 'y':
                print(f"Época {epoch+1}/{epochs}, Erro de Treino MSE: {mse_treino:.4f}, Erro de Validação MSE: {mse_validacao:.4f}, Acurácia de Validação: {acuracia_validacao:.4f}")

            # Parada antecipada
            if mse_validacao < melhor_erro_validacao:
                melhor_erro_validacao = mse_validacao
                epochs_erro_melhorou = 0
            else:
                epochs_erro_melhorou += 1

            if epochs_erro_melhorou >= early_stopping_limit:
                if verbose == 'y':
                    print(f"Parada antecipada na época {epoch+1} devido ao aumento do erro na validação.")
                break

    def prever(self, X):
        return self.forward(X)

    def mostrar_pesos(self,my_string):
        print("####################################################################################################################################")
        print(f'{my_string} :')
        print()
        for i, (peso, bias) in enumerate(zip(self.pesos, self.biases)):
            if i < len(self.pesos) - 1:
                # Camadas intermediárias
                print(f"\nCamada {i + 1}:")
                print(f"  Matriz de pesos:")
                print(peso)
                print(f"  Biases: ")
                print(bias[0])
            else:
                # Última camada (Camada de saída)
                print(f"\nCamada de Saída:")
                print(f"  Matriz de pesos:")
                print(peso)
                print(f"  Biases:")
                print(bias[0])

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
