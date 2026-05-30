import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def exibir_distribuicao_classes(y, label_encoder, titulo):
    """
    Exibe a quantidade e o percentual de registros por classe.
    """

    y_texto = label_encoder.inverse_transform(y)

    distribuicao = pd.Series(y_texto).value_counts()
    percentual = pd.Series(y_texto).value_counts(normalize=True) * 100

    relatorio = pd.DataFrame({
        "quantidade": distribuicao,
        "percentual": percentual.round(2)
    })

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print(relatorio)


def separar_treino_teste(df, coluna_entrada="review_text", coluna_alvo="target"):
    """
    Separa os dados em treino e teste.

    O stratify mantém a proporção das classes nos dois conjuntos.
    Isso é importante porque o dataset é desbalanceado.
    """

    X = df[coluna_entrada]
    y = df[coluna_alvo]

    label_encoder = LabelEncoder()
    y_codificado = label_encoder.fit_transform(y)

    print("\nMapeamento das classes:")
    for codigo, classe in enumerate(label_encoder.classes_):
        print(f"{codigo} -> {classe}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_codificado,
        test_size=0.2,
        random_state=42,
        stratify=y_codificado
    )

    print("\nSeparação treino/teste concluída.")
    print(f"Total geral: {len(X)} registros")
    print(f"Treino: {len(X_train)} registros")
    print(f"Teste: {len(X_test)} registros")

    percentual_treino = (len(X_train) / len(X)) * 100
    percentual_teste = (len(X_test) / len(X)) * 100

    print(f"Percentual treino: {percentual_treino:.2f}%")
    print(f"Percentual teste: {percentual_teste:.2f}%")

    exibir_distribuicao_classes(
        y_train,
        label_encoder,
        "DISTRIBUIÇÃO DAS CLASSES NO TREINO"
    )

    exibir_distribuicao_classes(
        y_test,
        label_encoder,
        "DISTRIBUIÇÃO DAS CLASSES NO TESTE"
    )

    return X_train, X_test, y_train, y_test, label_encoder