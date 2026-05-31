# pyrefly: ignore [missing-import]
import random
from pathlib import Path
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

# ======================================================================================== #
# Responsável por guarda funções úteis que ajudam o projeto a funcionar melhor
# ======================================================================================== #

def definir_semente(random_seed: int = 42) -> None:
    """
    Define uma seed para tentar manter os resultados reproduzíveis.

    Em redes neurais, alguns processos usam aleatoriedade, como:
        - inicialização dos pesos do modelo;
        - embaralhamento dos dados de treino;
        - algumas operações em GPU.

    Ao definir uma seed fixa, tentamos fazer com que o resultado seja
    mais parecido toda vez que o código for executado.
    """

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    # Essas configurações ajudam na reprodutibilidade em GPU.
    # Porém, podem deixar o treinamento um pouco mais lento.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def obter_dispositivo() -> torch.device:
    """
    Verifica se existe GPU CUDA disponível.

    Se existir GPU:
        retorna cuda

    Se não existir GPU:
        retorna cpu
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def criar_diretorios_necessarios() -> None:
    """
    Cria as pastas usadas pelo projeto, caso ainda não existam.

    Pastas criadas:
        - data
        - checkpoints
        - runs
    """

    diretorios = [
        Path("data"),
        Path("checkpoints"),
        Path("runs"),
    ]

    for diretorio in diretorios:
        diretorio.mkdir(parents=True, exist_ok=True)


def exibir_informacoes_dispositivo(dispositivo: torch.device) -> None:
    """
    Exibe informações sobre o dispositivo usado no treinamento.

    Isso ajuda a saber se o código está rodando em CPU ou GPU.
    """

    print(f"Dispositivo usado: {dispositivo}")

    if dispositivo.type == "cuda":
        print(f"Nome da GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU CUDA não disponível. O treinamento será executado na CPU.")