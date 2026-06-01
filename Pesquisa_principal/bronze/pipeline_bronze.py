# ======================================================================================
# pipeline.py
# ======================================================================================
# Responsabilidade:
# - Executar o fluxo da camada bronze
# - Transformar XLSX em CSV
# - Exibir informações iniciais da base
# ======================================================================================
# pyrefly: ignore [missing-import]
from Pesquisa_principal.constants import ARQUIVO_OUTPUT_BRONZE

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.transformacao_dados_xlsx_csv import transformar_base_feminicidio

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.informacoes_iniciais import diagnostico_inicial

# pyrefly: ignore [missing-import]
from Pesquisa_principal.bronze.base.utils import OutputTerminalEArquivo
from Pesquisa_principal.bronze.base.limpeza_dados import limpar_dados_bronze

def pipeline_bronze() -> None:
    """
    Executa a pipeline da camada bronze.

    Fluxo:
    1. Carrega o CSV existente ou transforma o XLSX em CSV
    2. Exibe o diagnóstico inicial da base bruta
    3. Executa a limpeza inicial dos dados
    """

    with OutputTerminalEArquivo(ARQUIVO_OUTPUT_BRONZE):
        print("=" * 80)
        print("INÍCIO DA PIPELINE BRONZE")
        print("=" * 80)

        dataframe = transformar_base_feminicidio()

        diagnostico_inicial(dataframe)

        dataframe_limpo = limpar_dados_bronze(dataframe)

        print("\n" + "=" * 80)
        print("FIM DA PIPELINE BRONZE")
        print("=" * 80)
        print(f"Output salvo em: {ARQUIVO_OUTPUT_BRONZE}")
        