import pandas as pd

def carregar_dataset(caminho_arquivo: str) -> pd.DataFrame:
    """
    Carrega o dataset a partir de um arquivo CSV.
    """

    df = pd.read_csv(caminho_arquivo)

    print("Dataset carregado com sucesso.")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    return df