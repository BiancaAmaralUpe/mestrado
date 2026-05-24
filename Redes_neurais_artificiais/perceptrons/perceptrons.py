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
    def __init__(self, learning_rate=0.1, n_epochs=10, pesos_iniciais=None, vies_inicial=0):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # pesos_iniciais --> permite definir uma estratégia diferente de inicialização
        # para cada caso, como AND, OR e XOR.
        self.pesos_iniciais = pesos_iniciais

        # vies_inicial --> permite definir o valor inicial do viés para cada caso.
        self.vies_inicial = vies_inicial

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

        if self.pesos_iniciais is None:
            self.pesos = np.zeros(n_colunas)
        else:
            self.pesos = np.array(self.pesos_iniciais, dtype=float)

        self.vies = self.vies_inicial

        for epoch in range(self.n_epochs):
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