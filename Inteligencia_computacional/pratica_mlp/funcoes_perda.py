# pyrefly: ignore [missing-import]
import torch.nn as nn

def criar_funcao_perda(tipo_loss: str):
    """
    Cria a função de perda usada no treinamento.

    cross_entropy:
        Função padrão para classificação multiclasse.
        Recebe logits brutos e rótulos inteiros.

    multi_margin:
        Função alternativa baseada em margem.
        Também recebe scores/logits e rótulos inteiros.
    """

    if tipo_loss == "cross_entropy":
        return nn.CrossEntropyLoss()

    if tipo_loss == "multi_margin":
        return nn.MultiMarginLoss()

    raise ValueError(f"Tipo de loss não reconhecido: {tipo_loss}")