# pyrefly: ignore [missing-import]
from .graficos_dados import mostrar_exemplos_mnist
# ========================================================================================
# Visualização inicial dos dados.
# ========================================================================================

def executar_visualizacao_dados(
    train_loader,
    quantidade_imagens: int = 16,
):
    """
    Mostra exemplos iniciais do dataset MNIST.
    """

    print("\nMostrando exemplos do MNIST")

    mostrar_exemplos_mnist(
        data_loader=train_loader,
        quantidade_imagens=quantidade_imagens,
    )