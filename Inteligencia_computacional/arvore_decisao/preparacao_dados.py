# preparacao_dados.py

import pandas as pd
from sklearn.model_selection import train_test_split


def separar_treino_teste(df: pd.DataFrame):
    """
    Separa o dataset em variáveis preditoras, variável alvo,
    conjunto de treino e conjunto de teste.
    """

    print("\n" + "=" * 70)
    print("PREPARAÇÃO DOS DADOS PARA MODELAGEM")
    print("=" * 70)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print("\nVariáveis preditoras utilizadas:")
    for coluna in X.columns:
        print(f"- {coluna}")

    print("\nVariável alvo:")
    print("- Outcome")

    print("\n" + "=" * 70)
    print("DIVISÃO ENTRE TREINO E TESTE")
    print("=" * 70)
    # ================================================================================= #
    # Definição inicial de treino e teste
    # test_size = 0.3 -> 30% dos dados serão usados para teste
    # random_state = 42 -> define a semente para geração de números aleatórios
    #                      garantindo que a divisão seja a mesma em todas as execuções
    # ================================================================================= #
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    print(f"Amostras para treino: {X_train.shape[0]}")
    print(f"Amostras para teste: {X_test.shape[0]}")
    print(f"Percentual de treino: 70%")
    print(f"Percentual de teste: 30%")

    return X_train, X_test, y_train, y_test