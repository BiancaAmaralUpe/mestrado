
# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import pandas as pd

# pyrefly: ignore [missing-import]
from sklearn.metrics import accuracy_score, confusion_matrix

# pyrefly: ignore [missing-import]
from .constants import (
    NOME_COLUNA_X1,
    NOME_COLUNA_X2,
    NOME_COLUNA_ALVO,
    NOMES_CLASSES,
)

# ======================================================================================== #
# Responsável por avaliar e exibir os resultados dos modelos no problema XOR.
#
# Este arquivo contém funções para:
#   - calcular acurácia;
#   - gerar matriz de confusão;
#   - criar tabelas de predições;
#   - comparar Perceptron simples e MLP;
#   - exibir a conclusão final da atividade.
# ======================================================================================== #

# ======================================================================================== #
# Calcula a acurácia do modelo.
#
# A acurácia mede a proporção de acertos entre os rótulos reais e os rótulos previstos.
# ======================================================================================== #
def avaliar_predicoes(y_real, y_previsto) -> float:
    """
    Calcula a acurácia das predições.

    A acurácia indica a proporção de acertos do modelo.

    Exemplo:
        Se o modelo acertou 3 de 4 exemplos:
            acurácia = 0.75
    """

    acuracia = accuracy_score(y_real, y_previsto)

    return acuracia

# ======================================================================================== #
# Gera a matriz de confusão para o problema XOR.
#
# A matriz de confusão mostra onde o modelo acertou e onde errou.
# Como o XOR possui duas classes, 0 e 1, a matriz terá formato 2x2.
# ======================================================================================== #
def gerar_matriz_confusao(y_real, y_previsto):
    """
    Gera a matriz de confusão.

    A matriz de confusão mostra:
        - quantos exemplos da classe 0 foram previstos como 0 ou 1;
        - quantos exemplos da classe 1 foram previstos como 0 ou 1.

    No XOR existem apenas duas classes:
        0 e 1.
    """

    matriz = confusion_matrix(
        y_real,
        y_previsto,
        labels=NOMES_CLASSES,
    )

    return matriz

# ======================================================================================== #
# Cria a tabela de resultados da MLP.
# ======================================================================================== #
def criar_tabela_resultados_perceptron(
    tabela_xor: pd.DataFrame,
    predicoes_perceptron,
) -> pd.DataFrame:
    """
    Cria uma tabela com os resultados do Perceptron simples.

    A tabela mostra:
        - entradas x1 e x2;
        - saída correta do XOR;
        - previsão do Perceptron;
        - se a previsão foi correta.
    """

    tabela_resultados = tabela_xor.copy()

    tabela_resultados["Perceptron prediction"] = predicoes_perceptron
    tabela_resultados["Correct?"] = (
        tabela_resultados[NOME_COLUNA_ALVO]
        == tabela_resultados["Perceptron prediction"]
    )

    return tabela_resultados

# ======================================================================================== #
# Exibe no terminal os resultados finais do Perceptron simples.
# Como o Perceptron simples é linear, espera-se que ele não atinja 100% de acurácia no XOR.
# ======================================================================================== #
def criar_tabela_resultados_mlp(
    tabela_xor: pd.DataFrame,
    probabilidades_mlp,
    predicoes_mlp,
) -> pd.DataFrame:
    """
    Cria uma tabela com os resultados da MLP.

    A tabela mostra:
        - entradas x1 e x2;
        - saída correta do XOR;
        - probabilidade estimada para a classe 1;
        - previsão da MLP;
        - se a previsão foi correta.
    """

    tabela_resultados = tabela_xor.copy()

    tabela_resultados["MLP probability class 1"] = np.round(
        probabilidades_mlp,
        4,
    )

    tabela_resultados["MLP prediction"] = predicoes_mlp
    tabela_resultados["Correct?"] = (
        tabela_resultados[NOME_COLUNA_ALVO]
        == tabela_resultados["MLP prediction"]
    )

    return tabela_resultados

# ======================================================================================== #
# Exibe a comparação final entre os dois modelos.
#
# Essa saída facilita a leitura do resultado principal:
#   - o Perceptron simples possui fronteira linear e falha no XOR;
#   - a MLP possui fronteira não linear e resolve o XOR.
# ======================================================================================== #
def criar_comparacao_final(
    acuracia_perceptron: float,
    acuracia_mlp: float,
) -> pd.DataFrame:
    """
    Cria uma tabela comparando Perceptron simples e MLP.

    A comparação mostra:
        - tipo de modelo;
        - tipo de fronteira de decisão;
        - acurácia obtida;
        - se resolveu o XOR perfeitamente.
    """

    comparacao = pd.DataFrame(
        [
            {
                "Modelo": "Perceptron simples",
                "Fronteira de decisão": "Uma linha reta",
                "Acurácia": acuracia_perceptron,
                "Resolve XOR perfeitamente?": "Não",
            },
            {
                "Modelo": "MLP",
                "Fronteira de decisão": "Fronteira não linear",
                "Acurácia": acuracia_mlp,
                "Resolve XOR perfeitamente?": "Sim",
            },
        ]
    )

    return comparacao

# ======================================================================================== #
# Exibe a conclusão conceitual da atividade.
#
# A conclusão explica por que:
#   - o Perceptron simples não resolve o XOR;
#   - a MLP consegue resolver o XOR;
#   - a camada oculta é importante para aprender transformações não lineares.
# ======================================================================================== #
def exibir_resultados_perceptron(
    pesos,
    bias,
    acuracia: float,
    tabela_resultados: pd.DataFrame,
) -> None:
    """
    Exibe os resultados finais do Perceptron simples.
    """

    print("\n" + "=" * 70)
    print("RESULTADOS DO PERCEPTRON SIMPLES")
    print("=" * 70)

    print(f"\nMelhores pesos encontrados: {pesos}")
    print(f"Melhor bias encontrado: {bias}")
    print(f"Acurácia do Perceptron: {acuracia:.4f}")

    print("\nTabela de predições do Perceptron:")
    print("-" * 70)
    print(tabela_resultados.to_string(index=False))

# ======================================================================================== #
# Responsável por avaliar os modelos treinados no problema XOR.
# ======================================================================================== #
def exibir_resultados_mlp(
    acuracia: float,
    tabela_resultados: pd.DataFrame,
) -> None:
    """
    Exibe os resultados finais da MLP.
    """

    print("\n" + "=" * 70)
    print("RESULTADOS DA MLP")
    print("=" * 70)

    print(f"\nAcurácia da MLP: {acuracia:.4f}")

    print("\nTabela de predições da MLP:")
    print("-" * 70)
    print(tabela_resultados.to_string(index=False))

# ======================================================================================== #
# Responsável por avaliar os modelos treinados no problema XOR.
# ======================================================================================== #
def exibir_comparacao_final(comparacao: pd.DataFrame) -> None:
    """
    Exibe a comparação final entre Perceptron simples e MLP.
    """

    print("\n" + "=" * 70)
    print("COMPARAÇÃO FINAL")
    print("=" * 70)

    print(comparacao.to_string(index=False))

# ======================================================================================== #
# Responsável por avaliar os modelos treinados no problema XOR.
# ======================================================================================== #
def exibir_conclusao_final(
    acuracia_perceptron: float,
    acuracia_mlp: float,
) -> None:
    """
    Exibe a conclusão final da atividade XOR.
    """

    print("\n" + "=" * 70)
    print("CONCLUSÃO")
    print("=" * 70)