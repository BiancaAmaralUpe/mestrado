import pandas as pd

from sklearn.utils import resample


def exibir_distribuicao_balanceamento(y, label_encoder, titulo):
    """
    Exibe a distribuição das classes antes ou depois do balanceamento.
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


def balancear_treino_oversampling(X_train, y_train, label_encoder):
    """
    Aplica Random Oversampling no conjunto de treino.

    A técnica replica aleatoriamente amostras das classes minoritárias
    até que todas as classes tenham a mesma quantidade de registros
    da classe majoritária.

    Importante:
    - O balanceamento deve ser aplicado apenas no treino.
    - O conjunto de teste deve permanecer com a distribuição original.
    """

    exibir_distribuicao_balanceamento(
        y_train,
        label_encoder,
        "DISTRIBUIÇÃO DO TREINO ANTES DO RANDOM OVERSAMPLING"
    )

    df_treino = pd.DataFrame({
        "review_text": X_train.values,
        "target": y_train
    })

    maior_classe = df_treino["target"].value_counts().max()

    datasets_balanceados = []

    for classe in sorted(df_treino["target"].unique()):
        df_classe = df_treino[df_treino["target"] == classe]

        df_classe_balanceada = resample(
            df_classe,
            replace=True,
            n_samples=maior_classe,
            random_state=42
        )

        datasets_balanceados.append(df_classe_balanceada)

    df_balanceado = pd.concat(datasets_balanceados)

    df_balanceado = df_balanceado.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    X_train_balanceado = df_balanceado["review_text"]
    y_train_balanceado = df_balanceado["target"].values

    exibir_distribuicao_balanceamento(
        y_train_balanceado,
        label_encoder,
        "DISTRIBUIÇÃO DO TREINO DEPOIS DO RANDOM OVERSAMPLING"
    )

    print("\nRandom Oversampling concluído.")
    print(f"Treino antes: {len(X_train)} registros")
    print(f"Treino depois: {len(X_train_balanceado)} registros")

    return X_train_balanceado, y_train_balanceado