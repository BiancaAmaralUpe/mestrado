# ========================================================================================
# Árvore Selvagem.
# ========================================================================================
# =================================================================================================================== #
# O Modelo entrou em overfithing, pois como ela não possue uma limitação, ela vai crescendo sem parar. Igual a uma raiz
# de arvore. Ela decorou 100% dos dados de treino. O modelo aprendeu os padrões muito especificos
# da base de treinamento (30%) e perdeu a capacidade de genalização para os novos dados
# A Árvore Selvagem não é uma boa candidata para produção, porque apresenta overfitting evidente.
# =================================================================================================================== #

# arvores_selvagens.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def treinar_arvore_selvagem(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print("========== ÁRVORE SELVAGEM ==============")
    print("=" * 70)

    print("\nConfiguração utilizada:")
    print("- criterion: gini")

    # ================================================
    # Não vai utilizar nenhum limitador. por isso que "none"
    # ================================================
    print("- max_depth: None")

    # ================================================
    # 
    # ================================================
    print("- random_state: 42")

    # ==============================================================================
    # Criação do modelo sem limite de profundidade
    # ==============================================================================
    modelo = DecisionTreeClassifier(
        random_state=42
    )

    # ==============================================================================
    # Treinamento do modelo
    # ==============================================================================
    modelo.fit(X_train, y_train)

    # ==============================================================================
    # Geração das previsões
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo das métricas de avaliação
    # ==============================================================================

    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    diferenca = acuracia_treino - acuracia_teste

    print("\nResultados:")
    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Diferença treino-teste: {diferenca * 100:.2f}%")

    return modelo
