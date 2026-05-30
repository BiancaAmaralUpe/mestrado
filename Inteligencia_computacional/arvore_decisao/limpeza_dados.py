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

    exibir_cabecalho("RESUMO GERAL DO DATASET")

    print(f"Total de linhas: {df.shape[0]}")
    print(f"Total de colunas: {df.shape[1]}")
    # ==============================================================================
    # Exibição das colunas disponíveis no dataset
    # Isso ajuda a confirmar quais variáveis estão disponíveis para o modelo
    # ==============================================================================
    print("\nColunas do dataset:")
    for coluna in df.columns:
        print(f"- {coluna}")

    print("\nPrimeiras linhas:")
    print(df.head().to_string(index=False))


def exibir_tipos_colunas(df: pd.DataFrame) -> None:
    """
    Exibe os tipos de dados de cada coluna.
    """

    exibir_cabecalho("TIPOS DAS COLUNAS")
    # ==============================================================================
    # Criação de uma tabela com os tipos de dados das colunas
    # Essa verificação ajuda a confirmar se as variáveis estão em formato adequado
    # ==============================================================================
    tipos = pd.DataFrame({
        "coluna": df.columns,
        "tipo": df.dtypes.astype(str).values
    })

    print(tipos.to_string(index=False))


def exibir_dados_ausentes(df: pd.DataFrame) -> None:
    """
    Exibe somente as colunas que possuem valores ausentes.
    """

    exibir_cabecalho("RELATÓRIO DE VALORES AUSENTES")
    # ==============================================================================
    # Cálculo da quantidade total de linhas e de valores ausentes
    # total_linhas =  guarda a quantidade de registros existentes no dataset.
    # total_ausentes =  calcula o total de valores ausentes em toda a base.
    # ==============================================================================
    total_linhas = len(df)
    total_ausentes = df.isna().sum().sum()

    print(f"Total geral de valores ausentes: {total_ausentes}")
    # ==============================================================================
    # Identificação das colunas que possuem valores ausentes
    # conta quantos valores ausentes existem em cada coluna
    # ==============================================================================
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
    # ========================================
    # Me mostra a quantidade de dados ausentes
    # ========================================
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

    exibir_cabecalho("RELATÓRIO DE LINHAS DUPLICADAS")
    # ==============================================================================
    # Verificação de linhas duplicadas no dataset
    # ==============================================================================
    total_duplicadas = df.duplicated().sum()

    print(f"Total de linhas duplicadas: {total_duplicadas}")

    if total_duplicadas == 0:
        print("\nNenhuma linha duplicada encontrada.")
    else:
        print("\nATENCAO: existem linhas duplicadas no dataset.")


def exibir_valores_zero_suspeitos(df: pd.DataFrame) -> None:
    """
    Exibe valores iguais a zero em colunas clínicas onde zero pode indicar inconsistência.
    """

    exibir_cabecalho("RELATÓRIO DE VALORES ZERO EM COLUNAS CLÍNICAS")

    colunas_zero_suspeito = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    total_linhas = len(df)
    registros = []
    # ==============================================================================
    # Contagem de valores zero em colunas clínicas sensíveis.
    # Percorre a lista de colunas onde o valor zero pode indicar um dado inconsistente
    # ou clinicamente improvável, como glicose, pressão arterial, insulina e IMC.
    # ==============================================================================
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

    print("\nObservação:")
    print(
        "Valores zero nessas colunas podem representar ausência de informação "
        "ou registros clinicamente improváveis, dependendo da interpretação da base."
    )


def exibir_estatisticas_descritivas(df: pd.DataFrame) -> None:
    """
    Exibe estatísticas descritivas do dataset.
    """

    exibir_cabecalho("ESTATÍSTICAS DESCRITIVAS")
    # ==============================================================================
    # Mostrar o describe do dataset
    # ==============================================================================
    print(df.describe().round(2).to_string())


def exibir_distribuicao_alvo(df: pd.DataFrame) -> None:
    """
    Exibe a distribuição da variável alvo Outcome.
    """

    exibir_cabecalho("DISTRIBUIÇÃO DA VARIÁVEL ALVO")
    # ==============================================================================
    # Verificação da existência da variável alvo
    # verificamos se a coluna "Outcome" existe.
    # Essa coluna é a variável alvo do problema, ou seja,
    # representa a classe que o modelo deverá prever.
    # ==============================================================================
    if "Outcome" not in df.columns:
        print("Coluna Outcome não encontrada.")
        return

    distribuicao = df["Outcome"].value_counts().sort_index()
    percentual = (df["Outcome"].value_counts(normalize=True).sort_index() * 100).round(2)

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
    Executa a verificação geral da qualidade dos dados.
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