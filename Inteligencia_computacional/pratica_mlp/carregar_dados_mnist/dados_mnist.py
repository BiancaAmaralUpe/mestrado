# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchvision
# pyrefly: ignore [missing-import]
import torchvision.transforms as transforms
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, random_split
# pyrefly: ignore [missing-import]
from ..constants import DATA_DIR
# ======================================================================================== #
# Responsável por carregar o MNIST e criar os DataLoaders
# ======================================================================================== #

def carregar_dados_mnist(
    tamanho_batch: int = 4096,
    percentual_validacao: float = 0.10,
    random_seed: int = 42,
):
    """
    Carrega o dataset MNIST e cria os DataLoaders de treino, validação e teste.

    O MNIST possui imagens de dígitos manuscritos de 0 a 9.
    """
    # ======================================================================================== #
    # Converte as imagens do MNIST para tensores do PyTorch.
    # ======================================================================================== #
    transformacao = transforms.ToTensor()
    
    # ======================================================================================== #
    # Carrega o conjunto completo de TREINAMENTO do MNIST.
    # ======================================================================================== #
    conjunto_treino_completo = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        transform=transformacao,
        download=True,
    )

    # ======================================================================================== #
    # Carrega o conjunto de TESTE do MNIST.
    # ======================================================================================== #
    conjunto_teste = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        transform=transformacao,
        download=True,
    )

    # ======================================================================================== #
    # Calcula a quantidade de imagens que será usada para VALIDACAO.
    # ======================================================================================== #
    quantidade_validacao = int(len(conjunto_treino_completo) * percentual_validacao)

    # ======================================================================================== #
    # Calcula a quantidade de imagens que continuará sendo usada para TREINO.
    # ======================================================================================== #
    quantidade_treino = len(conjunto_treino_completo) - quantidade_validacao
    
    # ======================================================================================== #
    # Divide o conjunto original de treino em TREINO e VALIDAÇÃO.
    # O que é random_split = divide o dataset de forma aleatoria 
    # ======================================================================================== #
    conjunto_treino, conjunto_validacao = random_split(
        conjunto_treino_completo,
        [quantidade_treino, quantidade_validacao],
        generator=torch.Generator().manual_seed(random_seed),
    )
    # ======================================================================================== #
    # pin_memory pode melhorar desempenho quando se usa GPU.
    # ======================================================================================== #
    usar_pin_memory = torch.cuda.is_available()
    
    # ======================================================================================== #
    # DataLoader de treino.
    # O que é pin_memory = Essa linha verifica se existe uma GPU CUDA disponível. pin_memory pode melhorar a velocidade de transferência dos dados da memória RAM para a GPU.
    # shuffle=True = dados serão embaralhados antes de cada época
    # ======================================================================================== #
    train_loader = DataLoader(
        conjunto_treino,
        batch_size=tamanho_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=usar_pin_memory,
    )
    # ======================================================================================== #
    # DataLoader de validação.
    # ======================================================================================== #
    val_loader = DataLoader(
        conjunto_validacao,
        batch_size=tamanho_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=usar_pin_memory,
    )

    # ======================================================================================== #
    # DataLoader de teste.
    # ======================================================================================== #
    test_loader = DataLoader(
        conjunto_teste,
        batch_size=tamanho_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=usar_pin_memory,
    )

    return train_loader, val_loader, test_loader

