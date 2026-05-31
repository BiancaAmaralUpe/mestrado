# arvore_solitaria.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

def treinar_arvore_solitaria(X_train, y_train):
    """
    Instancia e treina uma Árvore de Decisão sem limites.

    Ela representa o modelo baseline, ou seja, o modelo de comparação inicial.
    DecisionTreeClassifier(random_state=42)
    """

    print("\n" + "=" * 70)
    print("MODELO 1 - ÁRVORE SOLITÁRIA")
    print("=" * 70)

    # ==============================================================================
    # Criação do modelo de Árvore de Decisão.
    #
    # random_state=42:
    #   Garante que o modelo tenha o mesmo comportamento em todas as execuções.
    # sem restrições, justamente para observar o problema de overfitting.
    # ==============================================================================
    modelo = DecisionTreeClassifier(
        random_state=42
    )

    inicio = time.time()

    # ==============================================================================
    # Treinamento do modelo com os dados de treino.
    # ==============================================================================
    modelo.fit(X_train, y_train)

    fim = time.time()
    tempo_treinamento = fim - inicio

    print("Árvore Solitária treinada com sucesso.")
    print(f"Tempo de treinamento: {tempo_treinamento:.4f} segundos")

    return modelo, tempo_treinamento


def avaliar_arvore_solitaria(modelo, X_train, X_test, y_train, y_test):
    """
    Avalia a Árvore Solitária nos dados de treino e teste.

    Calcula:
    - Acurácia no treino
    - Acurácia no teste
    - Gap entre treino e teste

    Gap = acurácia treino - acurácia teste
    """

    print("\n" + "=" * 70)
    print("AVALIAÇÃO - ÁRVORE SOLITÁRIA")
    print("=" * 70)

    # ==============================================================================
    # O modelo faz previsões tanto no conjunto de treino quanto no conjunto de teste.
    #
    # A previsão no treino mostra o quanto o modelo aprendeu os dados usados
    # durante o treinamento.
    #
    # A previsão no teste mostra como o modelo se comporta diante de dados novos.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo da acurácia.
    #
    # Acurácia representa a proporção de previsões corretas feitas pelo modelo.
    # ==============================================================================
    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    # ==============================================================================
    # Cálculo do Gap.
    #
    # Um Gap alto indica que o modelo foi muito bem no treino,
    # mas perdeu desempenho no teste.
    #
    # Isso é um sinal de overfitting.
    # ==============================================================================
    gap = acuracia_treino - acuracia_teste

    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Gap treino-teste: {gap * 100:.2f}%")

    return {
        "modelo": "Árvore Solitária",
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
        "gap": gap
    }