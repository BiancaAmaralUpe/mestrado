# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
from sklearn.metrics import ConfusionMatrixDisplay

# pyrefly: ignore [missing-import]
from sklearn.decomposition import PCA

# pyrefly: ignore [missing-import]
from .constants import (
    OUTPUT_IMAGENS_DIR,
    RESOLUCAO_MALHA_GRAFICO,
    LIMITE_MINIMO_GRAFICO,
    LIMITE_MAXIMO_GRAFICO,
)


# ======================================================================================== #
# Responsável por gerar e salvar os gráficos da prática XOR.
#
# Os gráficos ajudam a visualizar:
#   - a distribuição dos pontos do XOR;
#   - a limitação do Perceptron simples;
#   - a fronteira não linear aprendida pela MLP;
#   - as matrizes de confusão;
#   - a transformação feita pela camada oculta da MLP.
# ======================================================================================== #

def obter_diretorio_saida(diretorio_saida=None):
    """
    Define onde as imagens serão salvas.

    Se nenhum diretório for informado, usa OUTPUT_IMAGENS_DIR.
    """

    if diretorio_saida is None:
        diretorio_saida = OUTPUT_IMAGENS_DIR

    diretorio_saida.mkdir(parents=True, exist_ok=True)

    return diretorio_saida


# ======================================================================================== #
# Plota os quatro pontos do problema XOR.
#
# Esse gráfico mostra a principal dificuldade do XOR:
#   - classe 0 nos pontos (0,0) e (1,1);
#   - classe 1 nos pontos (0,1) e (1,0).
#
# Como as classes ficam em diagonais opostas, uma única linha reta não consegue
# separar perfeitamente as duas classes.
# ======================================================================================== #
def plotar_pontos_xor(
    X,
    y,
    titulo: str = "Dataset XOR",
    diretorio_saida=None,
):
    """
    Plota os quatro pontos do problema XOR no espaço original.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    plt.figure(figsize=(6, 5))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=220,
        edgecolors="black",
    )

    for indice, (x1, x2) in enumerate(X):
        plt.text(
            x1 + 0.03,
            x2 + 0.03,
            f"class {y[indice]}",
            fontsize=11,
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(titulo)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.3, 1.3)
    plt.grid(True, alpha=0.3)

    caminho_imagem = diretorio_saida / "pontos_xor.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


# ======================================================================================== #
# Plota a acurácia do Perceptron ao longo das épocas.
#
# Esse gráfico mostra que o Perceptron simples não converge para 100% no XOR.
# Isso acontece porque o XOR não é linearmente separável.
# ======================================================================================== #
def plotar_historico_perceptron(
    historico_df,
    diretorio_saida=None,
):
    """
    Plota a acurácia do Perceptron simples durante o treinamento.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    plt.figure(figsize=(7, 4))

    plt.plot(
        historico_df["epoch"],
        historico_df["accuracy_after_epoch"],
        marker="o",
    )

    plt.xlabel("Época")
    plt.ylabel("Acurácia de treinamento")
    plt.title("Acurácia do Perceptron simples ao longo das épocas")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)

    caminho_imagem = diretorio_saida / "historico_perceptron.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


# ======================================================================================== #
# Plota a fronteira de decisão de um modelo no problema XOR.
#
# Essa função é genérica e serve tanto para:
#   - Perceptron simples;
#   - MLP.
#
# A ideia é criar uma malha de pontos no espaço x1 x x2, pedir ao modelo uma predição
# para cada ponto da malha e colorir as regiões previstas como classe 0 ou classe 1.
# ======================================================================================== #
def plotar_fronteira_decisao(
    X,
    y,
    funcao_predicao,
    titulo: str,
    nome_arquivo: str,
    funcao_score=None,
    threshold: float = 0.0,
    diretorio_saida=None,
):
    """
    Plota as regiões de decisão e a fronteira de decisão de um modelo.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    x_min = LIMITE_MINIMO_GRAFICO
    x_max = LIMITE_MAXIMO_GRAFICO
    y_min = LIMITE_MINIMO_GRAFICO
    y_max = LIMITE_MAXIMO_GRAFICO

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, RESOLUCAO_MALHA_GRAFICO),
        np.linspace(y_min, y_max, RESOLUCAO_MALHA_GRAFICO),
    )

    malha_pontos = np.c_[xx.ravel(), yy.ravel()]

    # Predição de classe para cada ponto da malha.
    regioes = funcao_predicao(malha_pontos).reshape(xx.shape)

    plt.figure(figsize=(7, 6))

    plt.contourf(
        xx,
        yy,
        regioes,
        alpha=0.25,
        cmap="coolwarm",
    )

    # Se existir score contínuo, desenha a fronteira exata.
    if funcao_score is not None:
        scores = funcao_score(malha_pontos).reshape(xx.shape)

        plt.contour(
            xx,
            yy,
            scores,
            levels=[threshold],
            colors="black",
            linewidths=2,
        )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=220,
        edgecolors="black",
    )

    for indice, (x1, x2) in enumerate(X):
        plt.text(
            x1 + 0.03,
            x2 + 0.03,
            f"class {y[indice]}",
            fontsize=11,
        )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(titulo)
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True, alpha=0.3)

    caminho_imagem = diretorio_saida / nome_arquivo
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


# ======================================================================================== #
# Plota a matriz de confusão.
#
# A matriz de confusão ajuda a comparar os erros do Perceptron simples
# com os resultados da MLP.
# ======================================================================================== #
def plotar_matriz_confusao(
    matriz_confusao,
    titulo: str,
    nome_arquivo: str,
    diretorio_saida=None,
):
    """
    Plota e salva uma matriz de confusão.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz_confusao,
        display_labels=[0, 1],
    )

    display.plot()

    plt.title(titulo)

    caminho_imagem = diretorio_saida / nome_arquivo
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()


# ======================================================================================== #
# Plota a transformação feita pela camada oculta da MLP.
#
# A camada oculta gera novas representações internas para os pontos do XOR.
# Como essa representação pode ter mais de duas dimensões, usamos PCA apenas
# para visualizar essas ativações em 2D.
# ======================================================================================== #
def plotar_representacao_oculta_mlp(
    X,
    y,
    modelo_mlp,
    diretorio_saida=None,
):
    """
    Plota a representação dos pontos XOR após a transformação da camada oculta da MLP.

    A MLP usada possui uma camada oculta. Para visualizar essa camada:
        1. Calculamos a saída linear da camada oculta.
        2. Aplicamos tanh, que é a ativação usada pela MLP.
        3. Aplicamos PCA para reduzir a representação para 2 dimensões.
    """

    diretorio_saida = obter_diretorio_saida(diretorio_saida)
    
    camada_oculta_linear = X @ modelo_mlp.coefs_[0] + modelo_mlp.intercepts_[0]
    ativacoes_ocultas = np.tanh(camada_oculta_linear)

    pca = PCA(n_components=2)
    representacao_2d = pca.fit_transform(ativacoes_ocultas)

    plt.figure(figsize=(6, 5))

    plt.scatter(
        representacao_2d[:, 0],
        representacao_2d[:, 1],
        c=y,
        cmap="coolwarm",
        s=220,
        edgecolors="black",
    )

    for indice, (h1, h2) in enumerate(representacao_2d):
        plt.text(
            h1 + 0.02,
            h2 + 0.02,
            f"class {y[indice]}",
            fontsize=11,
        )

    plt.xlabel("Representação oculta - componente PCA 1")
    plt.ylabel("Representação oculta - componente PCA 2")
    plt.title("Pontos XOR após transformação da camada oculta da MLP")
    plt.grid(True, alpha=0.3)

    caminho_imagem = diretorio_saida / "representacao_oculta_mlp.png"
    plt.savefig(caminho_imagem, dpi=300, bbox_inches="tight")
    print(f"Imagem salva em: {caminho_imagem}")

    plt.show()
    plt.close()