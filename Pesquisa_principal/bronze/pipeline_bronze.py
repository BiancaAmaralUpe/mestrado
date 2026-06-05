# ======================================================================================
# pipeline.py
# ======================================================================================
# Responsabilidade:
# - Executar o fluxo da camada bronze
# - Transformar XLSX em CSV
# - Exibir informações iniciais da base
# ======================================================================================

# pyrefly: ignore [missing-import]
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BRONZE, ARQUIVO_OUTPUT_ANALISE_DADOS, ARQUIVO_CSV_FEMINICIDIO_LIMPO

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.utils import OutputTerminalEArquivo

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.informacoes_iniciais import diagnostico_inicial

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_de_dados.transformacao_dados_xlsx_csv import transformar_base_feminicidio

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.tratamento_de_dados.limpeza_dados import limpar_dados_bronze, salvar_dataframe_limpo

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.estatisticas_descritivas import gerar_estatisticas_descritivas

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.analise_exploratoria import executar_analise_exploratoria

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.analise_dados.definir_metodologia import definir_metodologia_analise

def pipeline_bronze() -> None:
    """
    Executa a camada bronze.

    Fluxo:
    1. Executa transformação e limpeza estrutural
    2. Salva output_bronze.txt
    3. Executa análises da base limpa
    4. Salva output_analise.txt
    """

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_BRONZE):
        print("=" * 80)
        print("INÍCIO DA PIPELINE BRONZE - TRATAMENTO DOS DADOS")
        print("=" * 80)

        dataframe = transformar_base_feminicidio()

        diagnostico_inicial(dataframe)

        dataframe_limpo = limpar_dados_bronze(dataframe)

        salvar_dataframe_limpo(
            dataframe=dataframe_limpo,
            caminho_saida=ARQUIVO_CSV_FEMINICIDIO_LIMPO,
        )

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE BRONZE - TRATAMENTO DOS DADOS")
        print("=" * 80)
        print(f"Output bronze salvo em: {ARQUIVO_OUTPUT_BRONZE}")

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_ANALISE_DADOS):
        print("=" * 80)
        print("INÍCIO DA ANÁLISE DOS DADOS")
        print("=" * 80)

        gerar_estatisticas_descritivas(dataframe_limpo)

        executar_analise_exploratoria(dataframe_limpo)

        definir_metodologia_analise(dataframe_limpo)

        print("\n" + "=" * 80)
        print("FIM DA ANÁLISE DOS DADOS")
        print("=" * 80)
        print(f"Output da análise salvo em: {ARQUIVO_OUTPUT_ANALISE_DADOS}")