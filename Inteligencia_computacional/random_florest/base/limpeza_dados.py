# limpeza_dados.py

import pandas as pd

def exibir_cabecalho(titulo: str) -> None:
    """
    Exibe um cabeçalho visual no terminal.
    """

    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def exibir_resumo_dataset(df: pd.DataFrame) -> None:
    """
    Exibe informações gerais sobre o dataset.
    """

    exibir_cabecalho("===============RESUMO GERAL DO DATASET==================")

    print(f"Total de linhas: {df.shape[0]}")
    print(f"Total de colunas: {df.shape[1]}")

    print("\nColunas disponíveis no dataset:")
    for coluna in df.columns:
        print(f"- {coluna}")

    print("\nPrimeiras linhas do dataset:")
    print(df.head().to_string(index=False))


def exibir_tipos_colunas(df: pd.DataFrame) -> None:
    """
    Exibe os tipos de dados de cada coluna.
    """

    exibir_cabecalho("TIPOS DAS COLUNAS")

    tipos = pd.DataFrame({
        "coluna": df.columns,
        "tipo": df.dtypes.astype(str).values
    })

    print(tipos.to_string(index=False))


def exibir_dados_ausentes(df: pd.DataFrame) -> None:
    """
    Exibe a quantidade de valores ausentes por coluna.
    """

    exibir_cabecalho("==============RELATÓRIO DE VALORES AUSENTES==============")

    total_linhas = len(df)
    total_ausentes = df.isna().sum().sum()

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


def exibir_linhas_duplicadas(df: pd.DataFrame) -> None:
    """
    Exibe a quantidade de linhas duplicadas no dataset.
    """

    exibir_cabecalho("========================RELATÓRIO DE LINHAS DUPLICADAS=======================")

    total_duplicadas = df.duplicated().sum()

    print(f"Total de linhas duplicadas: {total_duplicadas}")

    if total_duplicadas == 0:
        print("\nNenhuma linha duplicada encontrada.")
    else:
        print("\nATENÇÃO: existem linhas duplicadas no dataset.")


def exibir_valores_zero_suspeitos(df: pd.DataFrame) -> None:
    """
    Exibe valores iguais a zero em colunas clínicas onde zero pode indicar inconsistência.
    """

    exibir_cabecalho("===========RELATÓRIO DE VALORES ZERO EM COLUNAS CLÍNICAS========")

    colunas_zero_suspeito = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    total_linhas = len(df)
    registros = []

    for coluna in colunas_zero_suspeito:
        if coluna in df.columns:
            qtd_zero = (df[coluna] == 0).sum()
            percentual_zero = round((qtd_zero / total_linhas) * 100, 2)

            registros.append({
                "coluna": coluna,
                "qtd_zeros": qtd_zero,
                "percentual_zeros": percentual_zero
            })

    relatorio = pd.DataFrame(registros)

    if relatorio.empty:
        print("Nenhuma coluna clínica esperada foi encontrada.")
        return

    print(relatorio.to_string(index=False))


def tratar_valores_zero_suspeitos(df: pd.DataFrame) -> pd.DataFrame:
    """
    No dataset de diabetes, algumas variáveis clínicas não deveriam assumir valor zero,
    como glicose, pressão arterial, espessura da pele, insulina e IMC.
    Por isso, esses zeros podem ser tratados como inconsistências.
    """

    exibir_cabecalho("==================TRATAMENTO DE VALORES ZERO SUSPEITOS================")

    df_tratado = df.copy()

    colunas_zero_suspeito = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for coluna in colunas_zero_suspeito:
        if coluna in df_tratado.columns:
            qtd_zero_antes = (df_tratado[coluna] == 0).sum()

            mediana = df_tratado.loc[df_tratado[coluna] != 0, coluna].median()

            df_tratado[coluna] = df_tratado[coluna].replace(0, mediana)

            qtd_zero_depois = (df_tratado[coluna] == 0).sum()

            print(f"Coluna: {coluna}")
            print(f"- Zeros antes: {qtd_zero_antes}")
            print(f"- Mediana utilizada: {mediana:.2f}")
            print(f"- Zeros depois: {qtd_zero_depois}")
            print("-" * 50)

    return df_tratado


def remover_linhas_duplicadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas duplicadas do dataset, caso existam.
    """

    exibir_cabecalho("=====================REMOÇÃO DE LINHAS DUPLICADAS========================")

    total_antes = len(df)
    df_tratado = df.drop_duplicates()
    total_depois = len(df_tratado)

    print(f"Total de linhas antes: {total_antes}")
    print(f"Total de linhas depois: {total_depois}")
    print(f"Linhas removidas: {total_antes - total_depois}")

    return df_tratado


def exibir_estatisticas_descritivas(df: pd.DataFrame) -> None:
    """
    Exibe estatísticas descritivas do dataset.
    """

    exibir_cabecalho("=================ESTATÍSTICAS DESCRITIVAS===============")

    print(df.describe().round(2).to_string())


def exibir_distribuicao_alvo(df: pd.DataFrame) -> None:
    """
    Exibe a distribuição da variável alvo Outcome.
    """

    exibir_cabecalho("=============DISTRIBUIÇÃO DA VARIÁVEL ALVO================")

    if "Outcome" not in df.columns:
        print("Coluna Outcome não encontrada.")
        return

    distribuicao = df["Outcome"].value_counts().sort_index()
    percentual = (
        df["Outcome"]
        .value_counts(normalize=True)
        .sort_index() * 100
    ).round(2)

    relatorio = pd.DataFrame({
        "classe": distribuicao.index,
        "quantidade": distribuicao.values,
        "percentual": percentual.values
    })

    print(relatorio.to_string(index=False))

    print("\nLegenda:")
    print("0 = Paciente sem indicação de diabetes")
    print("1 = Paciente com indicação de diabetes")


def verificar_dados(df: pd.DataFrame) -> None:
    """
    Executa a verificação geral da qualidade dos dados antes da limpeza.
    """

    exibir_cabecalho("VERIFICAÇÃO DA QUALIDADE DOS DADOS")

    exibir_resumo_dataset(df)
    exibir_tipos_colunas(df)
    exibir_dados_ausentes(df)
    exibir_linhas_duplicadas(df)
    exibir_valores_zero_suspeitos(df)
    exibir_estatisticas_descritivas(df)
    exibir_distribuicao_alvo(df)

    exibir_cabecalho("VERIFICAÇÃO CONCLUÍDA")
    print("Dataset verificado com sucesso.")


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executa a limpeza principal do dataset.

    Etapas:
    1. Remove linhas duplicadas.
    2. Trata valores zero suspeitos em colunas clínicas.
    3. Retorna um novo DataFrame limpo.
    """

    exibir_cabecalho("INÍCIO DA LIMPEZA DOS DADOS")

    df_limpo = df.copy()

    df_limpo = remover_linhas_duplicadas(df_limpo)
    df_limpo = tratar_valores_zero_suspeitos(df_limpo)

    exibir_cabecalho("LIMPEZA CONCLUÍDA")
    print("Dataset limpo com sucesso.")

    return df_limpo