# preparacao_dados.py

import pandas as pd
from sklearn.model_selection import train_test_split

def exibir_cabecalho(titulo: str) -> None:
    """
    Exibe um cabeçalho visual no terminal.
    """

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def separar_treino_teste(df: pd.DataFrame):
    """
    Separa o dataset em variáveis preditoras, variável alvo,
    conjunto de treino e conjunto de teste.

    Esta função segue exatamente o que foi solicitado no exercício:
    - test_size=0.3
    - random_state=42
    """

    exibir_cabecalho("PREPARAÇÃO DOS DADOS PARA MODELAGEM")

    if "Outcome" not in df.columns:
        raise ValueError("A coluna 'Outcome' não foi encontrada no dataset.")

    # ==============================================================================
    # Separação entre variáveis preditoras e variável alvo
    #
    # X = variáveis de entrada usadas pelo modelo para aprender os padrões
    # y = variável alvo que o modelo deverá prever
    #
    # No dataset de diabetes:
    # Outcome = 0 -> paciente sem indicação de diabetes
    # Outcome = 1 -> paciente com indicação de diabetes
    # ==============================================================================
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    print("\nVariáveis preditoras utilizadas:")
    for coluna in X.columns:
        print(f"- {coluna}")

    print("\nVariável alvo:")
    print("- Outcome")

    exibir_cabecalho("DIVISÃO ENTRE TREINO E TESTE")

    # ==============================================================================
    # Divisão dos dados conforme solicitado no exercício:
    #
    # test_size=0.3
    #   Define que 30% dos dados serão usados para teste.
    #
    # random_state=42
    #   Garante que a divisão seja sempre a mesma em todas as execuções.
    #   Isso permite comparar os modelos de forma justa.
    #
    # Importante:
    #   Não foi usado stratify porque o exercício pediu apenas test_size=0.3
    #   e random_state=42.
    # ==============================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    print(f"Amostras para treino: {X_train.shape[0]}")
    print(f"Amostras para teste: {X_test.shape[0]}")
    print("Percentual de treino: 70%")
    print("Percentual de teste: 30%")

    print("\nDistribuição da variável alvo no treino:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nDistribuição da variável alvo no teste:")
    print(y_test.value_counts().sort_index().to_string())

    return X_train, X_test, y_train, y_test