# normalizacao_dados.py
# =================================================================================== #
# 
# =================================================================================== #

import pandas as pd

def normalizar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para árvore de decisão, a normalização não é necessária.
    Esta função apenas retorna o DataFrame original.
    """

    print("\n[3] Normalização dos dados...")
    print("    -> Normalização não aplicada, pois DecisionTreeClassifier não exige escala padronizada.")

    return df