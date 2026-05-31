def exibir_estatisticas_descritivas(df: pd.DataFrame) -> None:
    """
    Exibe estatísticas descritivas do dataset.
    """

    exibir_cabecalho("=================ESTATÍSTICAS DESCRITIVAS===============")

    print(df.describe().round(2).to_string())
