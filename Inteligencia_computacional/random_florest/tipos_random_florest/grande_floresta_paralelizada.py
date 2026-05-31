# grande_floresta_paralelizada.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time


def treinar_grande_floresta_paralelizada(X_train, y_train):
    """
    Instancia e treina uma Random Forest com 150 árvores usando paralelização.

    Esta é a Grande Floresta Paralelizada do exercício.

    Conforme solicitado no enunciado:
    RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42)
    """

    print("\n" + "=" * 70)
    print("MODELO 3 - GRANDE FLORESTA PARALELIZADA")
    print("=" * 70)

    # ==============================================================================
    # Criação do modelo Random Forest com 150 árvores.
    #
    # n_estimators=150:
    #   Define que a floresta será composta por 150 árvores de decisão.
    #
    # n_jobs=-1:
    #   Diz que usar todos os núcleos disponíveis do processador.
    #   Isso permite treinar as árvores em paralelo, reduzindo o tempo de treinamento.
    #
    # random_state=42:
    #   Garante que os resultados sejam reproduzíveis em diferentes execuções.
    #
    # Não utilizamos max_depth aqui, porque o exercício pediu apenas:
    # n_estimators=150, n_jobs=-1 e random_state=42.
    # provavelmente vai dar overfiting
    # ==============================================================================
    modelo = RandomForestClassifier(
        n_estimators=150,
        n_jobs=-1,
        random_state=42
    )

    inicio = time.time()

    # ==============================================================================
    # Treinamento da grande floresta com os dados de treino.
    # ==============================================================================
    modelo.fit(X_train, y_train)

    fim = time.time()
    tempo_treinamento = fim - inicio

    print("Grande Floresta Paralelizada treinada com sucesso.")
    print(f"Tempo de treinamento: {tempo_treinamento:.4f} segundos")

    return modelo, tempo_treinamento

def avaliar_grande_floresta_paralelizada(modelo, X_train, X_test, y_train, y_test):
    """
    Avalia a Grande Floresta Paralelizada nos dados de treino e teste.

    Calcula:
    - Acurácia no treino
    - Acurácia no teste
    - Gap entre treino e teste

    Gap = acurácia treino - acurácia teste
    """

    print("\n" + "=" * 70)
    print("AVALIAÇÃO - GRANDE FLORESTA PARALELIZADA")
    print("=" * 70)

    # ==============================================================================
    # Previsões feitas pelo modelo nos dados de treino e teste.
    #
    # O treino mostra o desempenho nos dados que o modelo já viu.
    # O teste mostra o desempenho em dados novos.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo da acurácia para treino e teste.
    # ==============================================================================
    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    # ==============================================================================
    # Cálculo do Gap entre treino e teste.
    #
    # Um Gap muito alto pode indicar overfitting.
    # Um Gap menor indica que o modelo está generalizando melhor.
    # ==============================================================================
    gap = acuracia_treino - acuracia_teste

    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Gap treino-teste: {gap * 100:.2f}%")

    return {
        "modelo": "Grande Floresta Paralelizada",
        "acuracia_treino": acuracia_treino,
        "acuracia_teste": acuracia_teste,
        "gap": gap
    }
    