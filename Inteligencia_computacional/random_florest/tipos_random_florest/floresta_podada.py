# floresta_podada.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

def treinar_floresta_podada(X_train, y_train):
    """
    Instancia e treina uma Random Forest com 150 árvores e profundidade máxima 5.

    Esta é a Floresta Podada do exercício.

    Conforme solicitado no enunciado:
    RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )
    """

    print("\n" + "=" * 70)
    print("MODELO 4 - FLORESTA PODADA")
    print("=" * 70)

    # ==============================================================================
    # Criação do modelo Random Forest podado.
    #
    # n_estimators=150:
    #   Define que a floresta será composta por 150 árvores de decisão.
    #
    # max_depth=5:
    #   Limita a profundidade máxima de cada árvore a 5 níveis.
    #   Isso impede que as árvores cresçam demais e decorem os dados de treino.
    #
    # n_jobs=-1:
    #   Usa todos os núcleos disponíveis do processador para treinar as árvores
    #   em paralelo.
    #
    # random_state=42:
    #   Garante que os resultados sejam reproduzíveis em diferentes execuções.
    # ==============================================================================
    modelo = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        n_jobs=-1,
        random_state=42
    )

    inicio = time.time()

    # ==============================================================================
    # Treinamento da floresta podada com os dados de treino.
    # ==============================================================================
    modelo.fit(X_train, y_train)

    fim = time.time()
    tempo_treinamento = fim - inicio

    print("Floresta Podada treinada com sucesso.")
    print(f"Tempo de treinamento: {tempo_treinamento:.4f} segundos")

    return modelo, tempo_treinamento


def avaliar_floresta_podada(modelo, X_train, X_test, y_train, y_test):
    """
    Avalia a Floresta Podada nos dados de treino e teste.

    Calcula:
    - Acurácia no treino
    - Acurácia no teste
    - Gap entre treino e teste

    Gap = acurácia treino - acurácia teste
    """

    print("\n" + "=" * 70)
    print("AVALIAÇÃO - FLORESTA PODADA")
    print("=" * 70)

    # ==============================================================================
    # Previsões no conjunto de treino e no conjunto de teste.
    #
    # O conjunto de treino mostra o desempenho nos dados usados para aprendizado.
    # O conjunto de teste mostra a capacidade de generalização do modelo.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo da acurácia nos dois conjuntos.
    # ==============================================================================
    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    # ==============================================================================
    # Cálculo do Gap.
    #
    # Como esse modelo limita a profundidade das árvores, espera-se que o Gap
    # entre treino e teste seja menor do que em uma árvore sem poda.
    # ==============================================================================
    gap = acuracia_treino - acuracia_teste

    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Gap treino-teste: {gap * 100:.2f}%")

    return {
        "modelo": "Floresta Podada",
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
        "gap": gap
    }