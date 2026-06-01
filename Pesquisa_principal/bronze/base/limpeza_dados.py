# ======================================================================================
# limpeza_dados.py
# ======================================================================================
# Responsabilidade:
# - Remover linhas totalmente vazias
# - Padronizar nomes das colunas
# - Padronizar textos
# - Tratar valores textuais equivalentes a nulo
# - Gerar relatório simples da limpeza
# ======================================================================================

import pandas as pd


import pandas as pd


def validar_cabecalho(dataframe: pd.DataFrame) -> None:
    """
    Valida se o cabeçalho da base está correto.

    Se existirem colunas 'Unnamed', a base provavelmente foi carregada
    com o cabeçalho errado. Nesse caso, a correção deve acontecer na
    transformação XLSX -> CSV, e não removendo as colunas na limpeza.
    """

    colunas_unnamed = [
        coluna for coluna in dataframe.columns
        if str(coluna).startswith("Unnamed")
    ]

    if colunas_unnamed:
        raise ValueError(
            "Cabeçalho inválido detectado. Existem colunas 'Unnamed'. "
            "Reprocesse o CSV a partir do XLSX antes da limpeza. "
            f"Colunas problemáticas: {colunas_unnamed}"
        )

    if "Data da denúncia - Ano" not in dataframe.columns:
        raise ValueError(
            "Cabeçalho inválido detectado. A coluna esperada "
            "'Data da denúncia - Ano' não foi encontrada."
        )

    print("Cabeçalho validado com sucesso.")


def remover_linhas_totalmente_vazias(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas em que todos os valores são nulos.
    """

    qtd_antes = len(dataframe)

    dataframe = dataframe.dropna(how="all").copy()

    qtd_depois = len(dataframe)
    qtd_removidas = qtd_antes - qtd_depois

    print(f"Linhas totalmente vazias removidas: {qtd_removidas}")

    return dataframe


def padronizar_nomes_colunas(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas:
    - Remove espaços extras
    - Troca quebras de linha por espaço
    - Remove múltiplos espaços internos
    """

    dataframe = dataframe.copy()

    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
    )

    print("Nomes das colunas padronizados.")

    return dataframe


def padronizar_textos(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas de texto:
    - Remove espaços no início e no fim
    - Remove múltiplos espaços internos
    """

    dataframe = dataframe.copy()

    colunas_texto = dataframe.select_dtypes(include=["object", "string"]).columns

    for coluna in colunas_texto:
        dataframe[coluna] = (
            dataframe[coluna]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    print(f"Colunas textuais padronizadas: {len(colunas_texto)}")

    return dataframe


def converter_textos_nulos_para_na(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Converte textos que representam ausência de informação para pd.NA.

    Observação:
    - 'N/D' será mantido por enquanto, pois na sua base ele representa
      uma categoria informativa, não necessariamente um erro.
    """

    dataframe = dataframe.copy()

    valores_nulos_textuais = [
        "",
        "nan",
        "NaN",
        "none",
        "None",
        "NULL",
        "null",
    ]

    dataframe = dataframe.replace(valores_nulos_textuais, pd.NA)

    print("Textos equivalentes a nulo convertidos para pd.NA.")

    return dataframe


def relatorio_limpeza(dataframe: pd.DataFrame) -> None:
    """
    Exibe um relatório simples após a limpeza.
    """

    print("\n" + "=" * 80)
    print("RELATÓRIO APÓS LIMPEZA")
    print("=" * 80)

    print("\nQuantidade de linhas e colunas:")
    print(dataframe.shape)

    print("\nColunas da base após limpeza:")
    for indice, coluna in enumerate(dataframe.columns, start=1):
        print(f"{indice:02d}. {coluna}")

    print("\nTotal de valores nulos:")
    print(dataframe.isna().sum().sum())

    print("\nValores nulos por coluna:")
    print(dataframe.isna().sum())

    print("\nPrévia dos dados limpos:")
    print(dataframe.head())


def limpar_dados_bronze(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza inicial da camada bronze.

    Fluxo:
    1. Valida o cabeçalho
    2. Remove linhas totalmente vazias
    3. Padroniza nomes das colunas
    4. Padroniza textos
    5. Converte textos nulos para pd.NA
    6. Exibe relatório da limpeza
    """

    print("\n" + "=" * 80)
    print("INÍCIO DA LIMPEZA DOS DADOS - CAMADA BRONZE")
    print("=" * 80)

    validar_cabecalho(dataframe)

    dataframe = remover_linhas_totalmente_vazias(dataframe)
    dataframe = padronizar_nomes_colunas(dataframe)
    dataframe = padronizar_textos(dataframe)
    dataframe = converter_textos_nulos_para_na(dataframe)

    relatorio_limpeza(dataframe)

    print("\n" + "=" * 80)
    print("FIM DA LIMPEZA DOS DADOS - CAMADA BRONZE")
    print("=" * 80)

    return dataframe