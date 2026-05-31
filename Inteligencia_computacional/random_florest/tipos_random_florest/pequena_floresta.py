# pequena_floresta.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


def treinar_pequena_floresta(X_train, y_train):
    """
    Instancia e treina uma Random Forest com apenas 10 árvores.

    Esta é a Pequena Floresta do exercício.

    Conforme solicitado no enunciado:
    RandomForestClassifier(n_estimators=10, random_state=42)
    """

    print("\n" + "=" * 70)
    print("MODELO 2 - PEQUENA FLORESTA")
    print("=" * 70)

    # ==============================================================================
    # Criação do modelo Random Forest com 10 árvores.
    #
    # n_estimators=10:
    #   Define que a floresta será composta por 10 árvores de decisão.
    #
    # random_state=42:
    #   Garante que os resultados sejam reproduzíveis em diferentes execuções.
    #
    # Não utilizamos max_depth nem n_jobs aqui, porque o exercício pediu apenas
    # n_estimators=10 e random_state=42.
    # ==============================================================================
    modelo = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    inicio = time.time()

    # ==============================================================================
    # Treinamento da pequena floresta com os dados de treino.
    # ==============================================================================
    modelo.fit(X_train, y_train)

    fim = time.time()
    tempo_treinamento = fim - inicio

    print("Pequena Floresta treinada com sucesso.")
    print(f"Tempo de treinamento: {tempo_treinamento:.4f} segundos")

    return modelo, tempo_treinamento


def avaliar_pequena_floresta(modelo, X_train, X_test, y_train, y_test):
    """
    Avalia a Pequena Floresta nos dados de treino e teste.

    Calcula:
    - Acurácia no treino
    - Acurácia no teste
    - Gap entre treino e teste

    Gap = acurácia treino - acurácia teste
    """

    print("\n" + "=" * 70)
    print("AVALIAÇÃO - PEQUENA FLORESTA")
    print("=" * 70)

    # ==============================================================================
    # Previsões no conjunto de treino e no conjunto de teste.
    #
    # A previsão no treino ajuda a verificar o quanto o modelo aprendeu os dados.
    # A previsão no teste mostra o desempenho do modelo em dados que ele não viu.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo da acurácia em treino e teste.
    # ==============================================================================
    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    # ==============================================================================
    # Cálculo do Gap.
    #
    # Quanto menor o Gap, menor tende a ser a diferença entre o desempenho em treino
    # e o desempenho em teste.
    # ==============================================================================
    gap = acuracia_treino - acuracia_teste

    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Gap treino-teste: {gap * 100:.2f}%")

    return {
        "modelo": "Pequena Floresta",
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
        "gap": gap
    }