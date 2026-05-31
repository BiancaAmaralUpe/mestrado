# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import pandas as pd

# pyrefly: ignore [missing-import]
from .constants import (
    NOME_COLUNA_X1,
    NOME_COLUNA_X2,
    NOME_COLUNA_ALVO,
)

# ======================================================================================== #
# Responsável por criar e organizar os dados do problema XOR.
# ======================================================================================== #
def carregar_dados_xor():
    """
    Cria o dataset XOR.

    O XOR retorna:
        - 0 quando as duas entradas são iguais;
        - 1 quando as duas entradas são diferentes.

    Tabela esperada:
        x1  x2  XOR
        0   0   0
        0   1   1
        1   0   1
        1   1   0

    Retorna:
        X: matriz com as entradas do problema.
        y: vetor com os rótulos esperados.
    """

    # ======================================================================================== #
    # Entradas do problema XOR.
    #
    # Cada linha representa uma combinação possível entre x1 e x2.
    # ======================================================================================== #
    X = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )

    # ======================================================================================== #
    # Saídas esperadas do XOR.
    #
    # Resultado 0:
    #   quando x1 e x2 são iguais.
    #
    # Resultado 1:
    #   quando x1 e x2 são diferentes.
    # ======================================================================================== #
    y = np.array([0, 1, 1, 0])

    return X, y

# ======================================================================================== #
# Cria os dados de entrada e saída.
# ======================================================================================== #
def criar_tabela_xor(X, y):
    """
    Cria uma tabela em formato DataFrame para visualizar o dataset XOR.

    Retorna:
        DataFrame com as colunas:
            - x1
            - x2
            - XOR target
    """

    tabela_xor = pd.DataFrame(
        X,
        columns=[
            NOME_COLUNA_X1,
            NOME_COLUNA_X2,
        ],
    )

    tabela_xor[NOME_COLUNA_ALVO] = y

    return tabela_xor

# ======================================================================================== #
# Transforma os dados em uma tabela com pandas.
# ======================================================================================== #
def exibir_tabela_xor(tabela_xor) -> None:
    """
    Exibe a tabela XOR no terminal.
    """

    print("\nTabela do problema XOR")
    print("-" * 50)
    print(tabela_xor.to_string(index=False))