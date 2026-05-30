import pandas as pd
from pathlib import Path


def carregar_dataset(nome_arquivo: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir da pasta dados_diabetes.
    """

    pasta_atual = Path(__file__).resolve().parent
    caminho_arquivo = pasta_atual / "dados_diabetes" / nome_arquivo

    df = pd.read_csv(caminho_arquivo)

    print("\n[1] Dataset carregado com sucesso.")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    return df