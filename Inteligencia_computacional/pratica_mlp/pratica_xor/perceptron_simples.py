# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score


# ======================================================================================== #
# Implementação manual de um Perceptron simples para classificação binária.
#
# O objetivo é mostrar, de forma didática, como o Perceptron aprende:
#   - calcula uma combinação linear das entradas;
#   - aplica uma função degrau;
#   - compara a previsão com o valor real;
#   - atualiza pesos e bias quando erra.
#
# No problema XOR, o Perceptron simples não consegue chegar a 100% de acurácia,
# porque ele só cria uma fronteira de decisão linear.
# ======================================================================================== #


class PerceptronSimples:
    """
    Implementa um Perceptron simples para classificação binária.
    """

    def __init__(
        self,
        taxa_aprendizado: float = 1.0,
        quantidade_epocas: int = 20,
        ordem_amostras=None,
    ):
        self.taxa_aprendizado = taxa_aprendizado
        self.quantidade_epocas = quantidade_epocas
        self.ordem_amostras = ordem_amostras

    # ======================================================================================== #
    # Função degrau.
    #
    # Essa função transforma o valor linear z em uma classe binária.
    # Se z for maior ou igual a zero, retorna 1.
    # Caso contrário, retorna 0.
    # ======================================================================================== #
    def funcao_degrau(self, z):
        """
        Aplica a função degrau binária.
        """
        return np.where(z >= 0, 1, 0)

    # ======================================================================================== #
    # Calcula a entrada líquida do Perceptron.
    #
    # Essa é a parte linear do modelo:
    #   z = Xw + b
    #
    # X representa as entradas.
    # w representa os pesos.
    # b representa o bias.
    # ======================================================================================== #
    def calcular_entrada_linear(self, X):
        """
        Calcula a combinação linear entre entradas, pesos e bias.
        """
        return X @ self.pesos_ + self.bias_

    # ======================================================================================== #
    # Realiza a predição das classes.
    #
    # Primeiro calcula a entrada linear z.
    # Depois aplica a função degrau para transformar z em classe 0 ou 1.
    # ======================================================================================== #
    def prever(self, X):
        """
        Retorna as classes previstas pelo Perceptron.
        """

        entrada_linear = self.calcular_entrada_linear(X)
        predicoes = self.funcao_degrau(entrada_linear)

        return predicoes

    # ======================================================================================== #
    # Treina o Perceptron simples.
    #
    # Em cada época, o modelo percorre as amostras do XOR.
    # Para cada amostra:
    #   - faz uma previsão;
    #   - calcula o erro;
    #   - atualiza pesos e bias se houver erro.
    #
    # A regra de atualização é:
    #   pesos = pesos + taxa_aprendizado * erro * entrada
    #   bias  = bias  + taxa_aprendizado * erro
    # ======================================================================================== #
    def treinar(self, X, y):
        """
        Treina o Perceptron simples usando a regra clássica de atualização.

        Retorna:
            self:
                o próprio modelo treinado.
        """

        quantidade_amostras, quantidade_features = X.shape

        # ==================================================================================== #
        # Inicializa os pesos e o bias com zero.
        #
        # Como o XOR tem duas entradas, serão criados dois pesos:
        #   peso para x1
        #   peso para x2
        # ==================================================================================== #
        self.pesos_ = np.zeros(quantidade_features)
        self.bias_ = 0.0

        # ==================================================================================== #
        # Define a ordem de apresentação das amostras.
        #
        # Se nenhuma ordem for informada, usa a ordem natural dos dados.
        # ==================================================================================== #
        if self.ordem_amostras is None:
            self.ordem_amostras = list(range(quantidade_amostras))

        # ==================================================================================== #
        # Estruturas usadas para acompanhar o treinamento.
        #
        # historico_:
        #   guarda os resultados de cada época.
        #
        # melhor_acuracia_:
        #   guarda a maior acurácia encontrada.
        #
        # melhores_pesos_ e melhor_bias_:
        #   guardam os melhores parâmetros encontrados durante o treino.
        # ==================================================================================== #
        self.historico_ = []
        self.melhor_acuracia_ = -np.inf
        self.melhores_pesos_ = self.pesos_.copy()
        self.melhor_bias_ = self.bias_

        # ==================================================================================== #
        # Loop principal de treinamento.
        # ==================================================================================== #
        for epoca in range(1, self.quantidade_epocas + 1):
            erros_na_epoca = 0

            for indice_amostra in self.ordem_amostras:
                entrada = X[indice_amostra]
                rotulo_real = y[indice_amostra]

                predicao = self.prever(entrada.reshape(1, -1))[0]
                erro = rotulo_real - predicao

                # ============================================================================ #
                # Atualização dos pesos e do bias.
                #
                # Se o erro for zero, nada muda.
                # Se o erro for diferente de zero, o Perceptron ajusta os parâmetros.
                # ============================================================================ #
                self.pesos_ += self.taxa_aprendizado * erro * entrada
                self.bias_ += self.taxa_aprendizado * erro

                if erro != 0:
                    erros_na_epoca += 1

            # ================================================================================= #
            # Avalia a acurácia após terminar a época.
            # ================================================================================= #
            predicoes_epoca = self.prever(X)
            acuracia_epoca = accuracy_score(y, predicoes_epoca)

            # ================================================================================= #
            # Salva o histórico da época.
            # ================================================================================= #
            self.historico_.append(
                {
                    "epoch": epoca,
                    "errors_in_epoch": erros_na_epoca,
                    "accuracy_after_epoch": acuracia_epoca,
                    "weights": self.pesos_.copy(),
                    "bias": self.bias_,
                }
            )

            # ================================================================================= #
            # Guarda os melhores parâmetros encontrados.
            #
            # No XOR, o Perceptron simples não deve atingir 100%,
            # mas ainda podemos guardar a melhor solução linear encontrada.
            # ================================================================================= #
            if acuracia_epoca > self.melhor_acuracia_:
                self.melhor_acuracia_ = acuracia_epoca
                self.melhores_pesos_ = self.pesos_.copy()
                self.melhor_bias_ = self.bias_

        return self