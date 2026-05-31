# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import torch

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from .constants import OUTPUT_IMAGENS_DIR


# ======================================================================================== #
# Responsável somente pelos gráficos e visualizações.
# ======================================================================================== #

def obter_diretorio_saida(diretorio_saida=None):
    """
    Define onde as imagens serão salvas.

    Se um diretório específico for informado, usa esse diretório.
    Caso contrário, usa o diretório padrão OUTPUT_IMAGENS_DIR.
    """

    if diretorio_saida is None:
        diretorio_saida = OUTPUT_IMAGENS_DIR

    diretorio_saida.mkdir(parents=True, exist_ok=True)

    return diretorio_saida


def mostrar_exemplos_mnist(
    data_loader,
    quantidade_imagens: int = 16,
    diretorio_saida=None,
):
    """
    Mostra algumas imagens do dataset MNIST com seus respectivos rótulos reais.

    Essa função é útil antes do treinamento para confirmar se:
        - as imagens foram carregadas corretamente;
        - os rótulos estão corretos;
        - o formato dos dados faz sentido.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    imagens, rotulos = next(iter(data_loader))

    quantidade_imagens = min(quantidade_imagens, imagens.shape[0])

    quantidade_colunas = int(np.sqrt(quantidade_imagens))
    quantidade_linhas = int(np.ceil(quantidade_imagens / quantidade_colunas))

    plt.figure(figsize=(8, 8))

    for indice in range(quantidade_imagens):
        plt.subplot(quantidade_linhas, quantidade_colunas, indice + 1)

        # squeeze remove a dimensão do canal.
        # A imagem sai de [1, 28, 28] para [28, 28].
        plt.imshow(imagens[indice].squeeze(), cmap="gray")

        plt.title(f"Rótulo: {rotulos[indice].item()}")
        plt.axis("off")

    plt.suptitle("Exemplos do MNIST", fontsize=14)
    plt.tight_layout()

    caminho_imagem = diretorio_saida / "exemplos_mnist.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


def plotar_historico_treinamento(
    historico: dict,
    diretorio_saida=None,
):
    """
    Plota os gráficos de loss e acurácia durante o treinamento.

    O histórico deve conter:
        - train_loss
        - val_loss
        - train_accuracy
        - val_accuracy

    Esses gráficos ajudam a analisar:
        - se o modelo está aprendendo;
        - se a loss está diminuindo;
        - se a acurácia está aumentando;
        - se há indícios de overfitting.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    epocas = range(1, len(historico["train_loss"]) + 1)

    # ======================================================================================== #
    # Gráfico de loss.
    # ======================================================================================== #
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, historico["train_loss"], marker="o", label="Loss de treino")
    plt.plot(epocas, historico["val_loss"], marker="o", label="Loss de validação")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Loss da MLP durante o treinamento")
    plt.legend()
    plt.grid(True)

    caminho_imagem_loss = diretorio_saida / "historico_loss.png"
    plt.savefig(caminho_imagem_loss, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem_loss}")

    plt.show()
    plt.close()

    # ======================================================================================== #
    # Gráfico de acurácia.
    # ======================================================================================== #
    plt.figure(figsize=(8, 5))
    plt.plot(epocas, historico["train_accuracy"], marker="o", label="Acurácia de treino")
    plt.plot(epocas, historico["val_accuracy"], marker="o", label="Acurácia de validação")
    plt.xlabel("Época")
    plt.ylabel("Acurácia (%)")
    plt.title("Acurácia da MLP durante o treinamento")
    plt.legend()
    plt.grid(True)

    caminho_imagem_acuracia = diretorio_saida / "historico_acuracia.png"
    plt.savefig(caminho_imagem_acuracia, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem_acuracia}")

    plt.show()
    plt.close()


def plotar_matriz_confusao(
    matriz_confusao,
    nomes_classes,
    diretorio_saida=None,
):
    """
    Plota a matriz de confusão.

    A matriz de confusão mostra:
        - quais classes o modelo acertou;
        - quais classes o modelo confundiu.

    Linhas:
        representam os rótulos reais.

    Colunas:
        representam os rótulos previstos pelo modelo.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    plt.figure(figsize=(8, 7))
    plt.imshow(matriz_confusao, interpolation="nearest")
    plt.title("Matriz de Confusão - MLP")
    plt.colorbar()

    marcadores = np.arange(len(nomes_classes))

    plt.xticks(marcadores, nomes_classes)
    plt.yticks(marcadores, nomes_classes)

    plt.xlabel("Rótulo previsto")
    plt.ylabel("Rótulo real")

    limite_cor = matriz_confusao.max() / 2

    for linha in range(matriz_confusao.shape[0]):
        for coluna in range(matriz_confusao.shape[1]):
            valor = matriz_confusao[linha, coluna]

            cor_texto = "white" if valor > limite_cor else "black"

            plt.text(
                coluna,
                linha,
                str(valor),
                ha="center",
                va="center",
                color=cor_texto,
            )

    plt.tight_layout()

    caminho_imagem = diretorio_saida / "matriz_confusao.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


@torch.no_grad()
def mostrar_predicoes_modelo(
    modelo,
    data_loader,
    dispositivo: torch.device,
    quantidade_imagens: int = 16,
    diretorio_saida=None,
):
    """
    Mostra algumas imagens do conjunto de teste com:

        T: rótulo real
        P: rótulo previsto pelo modelo
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    modelo.eval()

    imagens, rotulos_reais = next(iter(data_loader))

    imagens_dispositivo = imagens.to(dispositivo)

    logits = modelo(imagens_dispositivo)
    rotulos_previstos = torch.argmax(logits, dim=1).cpu()

    quantidade_imagens = min(quantidade_imagens, imagens.shape[0])

    quantidade_colunas = int(np.sqrt(quantidade_imagens))
    quantidade_linhas = int(np.ceil(quantidade_imagens / quantidade_colunas))

    plt.figure(figsize=(8, 8))

    for indice in range(quantidade_imagens):
        rotulo_real = rotulos_reais[indice].item()
        rotulo_previsto = rotulos_previstos[indice].item()

        plt.subplot(quantidade_linhas, quantidade_colunas, indice + 1)

        plt.imshow(imagens[indice].squeeze(), cmap="gray")
        plt.title(f"T: {rotulo_real} | P: {rotulo_previsto}")
        plt.axis("off")

    plt.suptitle("Predições da MLP no conjunto de teste", fontsize=14)
    plt.tight_layout()

    caminho_imagem = diretorio_saida / "predicoes_modelo.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()