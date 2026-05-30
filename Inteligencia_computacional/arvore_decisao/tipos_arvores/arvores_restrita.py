# ================================================================================ #
#                                   ARVORES                                        #
#                                   RESTRITA                                       #
# ================================================================================ #
# ============================================================================================================== #
# A árvore restrita apresentou desempenho de 76,35% no treino e 71,86% no teste. 
# A diferença entre os dois conjuntos foi de 4,49 pontos percentuais, indicando que
# o modelo não apresentou um sobreajuste muito alto. 
# Como a profundidade foi limitada a 3 níveis, a árvore ficou mais simples e 
# com menor risco de memorizar os dados de treinamento. 
# Porém, essa limitação também pode reduzir um pouco a capacidade do modelo de capturar padrões mais complexos.
# Como ela possui uma limitação , a probalidade dela entrar em overfithing é menor.
# =============================================================================================================== #
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def treinar_arvore_restrita(X_train, X_test, y_train, y_test):
    """
    Treina e avalia uma Árvore de Decisão Restrita.
    Configuração: max_depth=3.
    """

    print("\n" + "=" * 70)
    print("==========ÁRVORE RESTRITA==============")
    print("=" * 70)

    print("\nConfiguração utilizada:")
    print("- criterion: gini")
    print("- max_depth: 3")
    print("- random_state: 42")
    # ================================================================================= #
    # Definindo a Árvore de Decisão Restrita
    # - criterion: gini ->mede a qualidade dos splits (usandogini)
    # - max_depth: 3 -> define a profundidade máxima da árvore
    # - random_state: 42 ->define a semente para geração de números aleatórios
    #                     garantindo que a divisão seja a mesma em todas as execuções
    # ================================================================================= #
    modelo = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=42
    )

    modelo.fit(X_train, y_train)

    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    # ================================================================================= #
    # avaliar o desempenho da árvore de decisão
    # Acurácia no treino = respostas reais do treino  x  respostas previstas pelo modelo no treino
    # accuracy_score calcula o percentual de acertos = Essa métrica mostra o quanto o modelo aprendeu nos dados que ele já viu.
    # ================================================================================= #
    acuracia_treino = accuracy_score(y_train, y_pred_train)
    # ================================================================================= #
    # Acurácia no teste
    # respostas reais do teste  x  respostas previstas pelo modelo no teste
    # conjunto de teste é formado por dados que o modelo não usou para aprender.
    # ================================================================================= #
    acuracia_teste = accuracy_score(y_test, y_pred_test)
    # ================================================================================= #
    # Essa linha mede a diferença entre o desempenho no treino e no teste. Ajuda a 
    # identificar possivel overfiting
    # ================================================================================= #
    diferenca = acuracia_treino - acuracia_teste

    print("\nResultados:")
    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Diferença treino-teste: {diferenca * 100:.2f}%")
    # ================================================================================= #
    # Retorna o modelo treinado para ser usado em outras etapas.
    # ================================================================================= # 
    return modelo
