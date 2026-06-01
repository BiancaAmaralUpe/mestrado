# ======================================================================================
# constants.py
# ======================================================================================
# Responsabilidade:
# - Centralizar caminhos e nomes de arquivos usados no projeto
# ======================================================================================

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

BRONZE_DIR = ROOT_DIR / "bronze"

DIRETORIO_DADOS_FEMINICIDIO = BRONZE_DIR / "dados_feminicidio"

ARQUIVO_XLSX_FEMINICIDIO = DIRETORIO_DADOS_FEMINICIDIO / "2020_primeiro_semestre.xlsx"
ARQUIVO_CSV_FEMINICIDIO = DIRETORIO_DADOS_FEMINICIDIO / "2020_primeiro_semestre.csv"

ARQUIVO_OUTPUT_BRONZE = DIRETORIO_DADOS_FEMINICIDIO / "output_bronze.txt"