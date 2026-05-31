# pyrefly: ignore [missing-import]
import pandas as pd

# pyrefly: ignore [missing-import]
from .constants import (
    TAXA_APRENDIZADO_PERCEPTRON,
    QUANTIDADE_EPOCAS_PERCEPTRON,
    ORDEM_AMOSTRAS_PERCEPTRON,
)

# pyrefly: ignore [missing-import]
from .perceptron_simples import PerceptronSimples

# pyrefly: ignore [missing-import]
from .modelo_mlp import criar_modelo_mlp

# ======================================================================================= #
# Responsável por treinar os modelos usados na prática XOR.
#
# Este arquivo contém funções para:
#   - treinar o Perceptron simples;
#   - treinar a MLP;
#   - organizar o histórico de treinamento do Perceptron em uma tabela.
# ======================================================================================== #


# ======================================================================================== #
# Treina o Perceptron simples.
#
# O Perceptron recebe os dados XOR e tenta encontrar uma reta capaz de separar
# as classes 0 e 1.
#
# Como o XOR não é linearmente separável, espera-se que o Perceptron simples
# não consiga atingir 100% de acurácia.
# ======================================================================================== #
def treinar_perceptron_simples(X, y) -> PerceptronSimples:
    """
    Treina o Perceptron simples no problema XOR.
    """

    print("\n" + "=" * 70)
    print("===================TREINANDO PERCEPTRON SIMPLES======================")
    print("=" * 70)

    # ==================================================================================== #
    # A ordem fixa das amostras ajuda a reproduzir o mesmo resultado em diferentes execuções.
    # ==================================================================================== #
    perceptron = PerceptronSimples(
        taxa_aprendizado=TAXA_APRENDIZADO_PERCEPTRON,
        quantidade_epocas=QUANTIDADE_EPOCAS_PERCEPTRON,
        ordem_amostras=ORDEM_AMOSTRAS_PERCEPTRON,
    )

    # ==================================================================================== #
    # Executa o treinamento manual do Perceptron.
    #
    # Durante o treino, o próprio objeto armazena:
    #   - histórico por época;
    #   - melhores pesos encontrados;
    #   - melhor bias encontrado;
    #   - melhor acurácia obtida.
    # ==================================================================================== #
    perceptron.treinar(X, y)

    print("Treinamento do Perceptron simples finalizado.")
    print(f"Melhor acurácia encontrada: {perceptron.melhor_acuracia_:.4f}")

    return perceptron


# ======================================================================================== #
# Converte o histórico do Perceptron para DataFrame.
#
# O histórico é salvo como lista de dicionários dentro do objeto Perceptron.
# Transformar em DataFrame facilita:
#   - visualizar as épocas;
#   - plotar a curva de acurácia;
#   - exibir os erros por época.
# ======================================================================================== #
def criar_historico_perceptron_df(perceptron: PerceptronSimples) -> pd.DataFrame:
    """
    Cria um DataFrame com o histórico de treinamento do Perceptron.
    """

    historico_df = pd.DataFrame(perceptron.historico_)

    return historico_df


# ======================================================================================== #
# Treina a MLP no problema XOR.
#
# Diferente do Perceptron simples, a MLP possui camada oculta com ativação não linear.
# Essa camada permite transformar o espaço de entrada e resolver o XOR.
# ======================================================================================== #
def treinar_mlp(X, y):
    """
    Cria e treina a MLP no problema XOR.
    """

    print("\n" + "=" * 70)
    print("======================TREINANDO MLP======================")
    print("=" * 70)

    # ==================================================================================== #
    # Cria a MLP definida em modelo_mlp.py.
    #
    # A arquitetura vem do constants.py:
    #   - camada oculta;
    #   - função de ativação;
    #   - solver;
    #   - número máximo de iterações.
    # ==================================================================================== #
    modelo_mlp = criar_modelo_mlp()

    # ==================================================================================== #
    # Treina a MLP.
    #
    # O método fit ajusta os pesos internos da rede para aprender a relação XOR.
    # Como a MLP possui camada oculta não linear, ela deve conseguir resolver
    # corretamente os quatro casos do XOR.
    # ==================================================================================== #
    modelo_mlp.fit(X, y)

    print("Treinamento da MLP finalizado.")

    return modelo_mlp