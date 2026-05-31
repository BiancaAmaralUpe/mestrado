# ======================================================================================
# transformacao_dado_xlsx_csv.py
# ======================================================================================
# Responsabilidade:
# - Ler um arquivo Excel (.xlsx)
# - Converter para CSV
# - Salvar o CSV em uma pasta de saída
# ======================================================================================

from pathlib import Path

import pandas as pd


def transformar_xlsx_em_csv(
    caminho_xlsx: str,
    caminho_csv: str,
    nome_aba: int | str = 0,
) -> pd.DataFrame:
    """
    Lê um arquivo Excel e salva uma versão em CSV.
    """

    caminho_xlsx = Path(caminho_xlsx)
    caminho_csv = Path(caminho_csv)

    if not caminho_xlsx.exists():
        raise FileNotFoundError(f"Arquivo Excel não encontrado: {caminho_xlsx}")

    print(f"Lendo arquivo Excel: {caminho_xlsx}")

    dataframe = pd.read_excel(caminho_xlsx, sheet_name=nome_aba)

    caminho_csv.parent.mkdir(parents=True, exist_ok=True)

    dataframe.to_csv(caminho_csv, index=False, encoding="utf-8")

    print(f"CSV salvo com sucesso em: {caminho_csv}")
    print(f"Linhas e colunas: {dataframe.shape}")

    return dataframe