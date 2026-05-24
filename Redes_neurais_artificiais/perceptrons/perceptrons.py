# pyrefly: ignore [missing-import]
import numpy as np

# perceptrons.py


class Perceptron:
    #============================================================================================#
    # Inicialização do Perceptron
    #
    # learning_rate --> controla o tamanho do ajuste que o modelo faz quando erra.
    # n_epochs      --> número de vezes que o modelo vai passar pelos dados para aprender.
    # pesos         --> são os pesos do modelo. Cada entrada possui um peso.
    # vies          --> valor extra que ajuda o modelo a tomar a decisão final.
    #============================================================================================#
    def __init__(self, learning_rate=0.1, n_epochs=10):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Começam como None porque ainda serão criados no treinamento.
        self.pesos = None
        self.vies = None

    #============================================================================================#
    # A função de ativação decide se a saída será 0 ou 1.
    #
    # Se o valor calculado for maior ou igual a zero, retorna 1.
    # Caso contrário, retorna 0.
    #============================================================================================#
    def funcao_ativacao(self, valor):
        return 1 if valor >= 0 else 0

    #============================================================================================#
    # Essa é a parte onde o Perceptron aprende.
    #============================================================================================#
    def fit(self, X, y):
        # X.shape retorna:
        # n_amostras -> quantidade de linhas/exemplos
        # n_colunas  -> quantidade de entradas em cada exemplo
        n_amostras, n_colunas = X.shape

        # Cria um peso para cada coluna de entrada.
        # No caso do AND, temos duas entradas, então teremos dois pesos.
        self.pesos = np.zeros(n_colunas)

        # O viés começa em zero.
        self.vies = 0

        # Repete o treinamento várias vezes.
        for epoch in range(self.n_epochs):

            # Passa por cada entrada do conjunto X.
            for indice, entrada_atual in enumerate(X):

                # ======================================================#
                # Calcula a saída linear:
                #
                # resultado_linear = entrada * pesos + vies
                #
                # No caso do AND:
                # resultado_linear = x1*peso1 + x2*peso2 + vies
                # ======================================================#
                resultado_linear = np.dot(entrada_atual, self.pesos) + self.vies

                # ======================================================#
                # Transforma o resultado da conta em 0 ou 1.
                # ======================================================#
                previsao = self.funcao_ativacao(resultado_linear)

                # ======================================================#
                # Calcula o erro:
                #
                # erro = resposta correta - resposta prevista
                # ======================================================#
                erro = y[indice] - previsao

                # ======================================================#
                # Atualiza os pesos.
                #
                # novo peso = peso antigo + taxa de aprendizado * erro * entrada
                # ======================================================#
                self.pesos += self.learning_rate * erro * entrada_atual

                # ======================================================#
                # Atualiza o viés.
                #
                # novo vies = vies antigo + taxa de aprendizado * erro
                # ======================================================#
                self.vies += self.learning_rate * erro

    #============================================================================================#
    # Essa é a parte onde o Perceptron faz previsões.
    #============================================================================================#
    def predict(self, X):
        previsoes = []

        # Passa por cada entrada.
        for entrada_atual in X:

            # ======================================================#
            # Faz a conta:
            #
            # resultado_linear = x1*peso1 + x2*peso2 + vies
            # ======================================================#
            resultado_linear = np.dot(entrada_atual, self.pesos) + self.vies

            # ======================================================#
            # Transforma o resultado da conta em 0 ou 1.
            # ======================================================#
            previsao = self.funcao_ativacao(resultado_linear)

            # ======================================================#
            # Guarda a previsão na lista.
            # ======================================================#
            previsoes.append(previsao)

        # Retorna a lista de previsões como array NumPy.
        return np.array(previsoes)