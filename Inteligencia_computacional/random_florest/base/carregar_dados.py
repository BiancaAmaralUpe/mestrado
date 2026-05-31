# carregar_dados.py

import pandas as pd
from pathlib import Path

def carregar_dataset(nome_arquivo: str) -> pd.DataFrame:
    """
    Carrega o dataset de diabetes a partir da pasta dados_diabetes.
    """

    # ==============================================================================
    # Localiza a pasta onde este arquivo carregar_dados.py está salvo.
    # ==============================================================================
    pasta_atual = Path(__file__).resolve().parent

    # ==============================================================================
    # Monta o caminho completo até o arquivo CSV.
    # ==============================================================================
    caminho_arquivo = pasta_atual.parent / "dados_diabetes" / nome_arquivo

    # ==============================================================================
    # Lê o arquivo CSV e transforma em um DataFrame do pandas.
    # ==============================================================================
    df = pd.read_csv(caminho_arquivo)

    print("\n" + "=" * 70)
    print("CARREGAMENTO DO DATASET")
    print("=" * 70)

    print("\nDataset carregado com sucesso.")
    print(f"Caminho do arquivo: {caminho_arquivo}")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    return df