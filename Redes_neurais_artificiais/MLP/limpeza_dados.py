import pandas as pd


def exibir_dados_ausentes(df: pd.DataFrame, titulo: str):
    """
    Exibe somente as colunas que possuem valores ausentes.
    Evita repetir colunas com 0 valores nulos.
    """

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)

    total_linhas = len(df)
    total_colunas = df.shape[1]
    total_ausentes = df.isna().sum().sum()

    print(f"Total de linhas: {total_linhas}")
    print(f"Total de colunas: {total_colunas}")
    print(f"Total geral de valores ausentes: {total_ausentes}")

    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]

    if ausentes.empty:
        print("\nNenhuma coluna com valores ausentes.")
        return

    relatorio = pd.DataFrame({
        "coluna": ausentes.index,
        "qtd_ausentes": ausentes.values,
        "percentual_ausentes": ((ausentes.values / total_linhas) * 100).round(2)
    })

    relatorio = relatorio.sort_values(
        by="qtd_ausentes",
        ascending=False
    )

    print("\nColunas com valores ausentes:")
    print(relatorio.to_string(index=False))


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa o dataset e cria a variável alvo para classificação.
    """

    df = df.copy()

    linhas_antes = len(df)
    colunas_antes = df.shape[1]

    exibir_dados_ausentes(
        df,
        "RELATÓRIO ANTES DA LIMPEZA"
    )

    colunas_remover = [
        "episodes_watched",
        "anime_title",
        "anime_mal_id",
        "review_id",
        "username",
        "date"
    ]

    colunas_existentes_removidas = [
        coluna for coluna in colunas_remover if coluna in df.columns
    ]

    df = df.drop(
        columns=colunas_remover,
        errors="ignore"
    )

    df = df.dropna(
        subset=[
            "review_text",
            "tags"
        ]
    )

    df["target"] = df["tags"].astype(str).str.split("|").str[0]

    classes_validas = [
        "Recommended",
        "Mixed Feelings",
        "Not Recommended"
    ]

    df = df[df["target"].isin(classes_validas)]

    df = df[df["review_text"].astype(str).str.strip() != ""]

    linhas_depois = len(df)
    colunas_depois = df.shape[1]

    exibir_dados_ausentes(
        df,
        "RELATÓRIO DEPOIS DA LIMPEZA"
    )

    print("\n" + "=" * 70)
    print("RESUMO DA LIMPEZA")
    print("=" * 70)
    print(f"Linhas antes: {linhas_antes}")
    print(f"Linhas depois: {linhas_depois}")
    print(f"Linhas removidas: {linhas_antes - linhas_depois}")

    print(f"Colunas antes: {colunas_antes}")
    print(f"Colunas depois: {colunas_depois}")
    print(f"Colunas efetivamente removidas: {len(colunas_existentes_removidas)}")
    print("Colunas adicionadas: 1")
    print("Coluna adicionada:")
    print("- target")

    print("\nColunas removidas:")
    for coluna in colunas_existentes_removidas:
        print(f"- {coluna}")

    print("\nDistribuição final das classes:")
    print(df["target"].value_counts())

    return df