# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import learning_curve


def plotar_curva_aprendizagem(
    modelo,
    X,
    y,
    titulo="Curva de aprendizagem da MLP"
):
    """
    Gera a curva de aprendizagem do modelo.

    A curva compara o desempenho no treino e na validação
    conforme aumenta a quantidade de dados usados no treinamento.
    """
    #=======================================================================#
    #                              
    #=======================================================================#
    print("\n" + "=" * 70)
    print("GERANDO CURVA DE APRENDIZAGEM")
    print("=" * 70)
    #=======================================================================#
    #                              
    #=======================================================================#
    tamanhos_treino, scores_treino, scores_validacao = learning_curve(
        estimator=modelo,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42
    )

    media_treino = scores_treino.mean(axis=1)
    desvio_treino = scores_treino.std(axis=1)

    media_validacao = scores_validacao.mean(axis=1)
    desvio_validacao = scores_validacao.std(axis=1)

    print("\nResultados da curva de aprendizagem:")
    for tamanho, treino, validacao in zip(
        tamanhos_treino,
        media_treino,
        media_validacao
    ):
        print(
            f"Tamanho treino: {tamanho} | "
            f"F1 treino: {treino:.4f} | "
            f"F1 validação: {validacao:.4f}"
        )

    plt.figure(figsize=(10, 6))

    plt.plot(
        tamanhos_treino,
        media_treino,
        marker="o",
        label="Treino"
    )

    plt.plot(
        tamanhos_treino,
        media_validacao,
        marker="o",
        label="Validação"
    )

    plt.fill_between(
        tamanhos_treino,
        media_treino - desvio_treino,
        media_treino + desvio_treino,
        alpha=0.2
    )

    plt.fill_between(
        tamanhos_treino,
        media_validacao - desvio_validacao,
        media_validacao + desvio_validacao,
        alpha=0.2
    )

    plt.title(titulo)
    plt.xlabel("Quantidade de dados de treino")
    plt.ylabel("F1-score macro")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()