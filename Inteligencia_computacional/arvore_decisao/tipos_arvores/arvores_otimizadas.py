# ================================================================================ #
#                                   ARVORES                                        #
#                                   OTIMIZADAS                                     #
# ================================================================================ #
# ============================================================================================================== #
# Foi definido que essa arvore tivesse um novo parametro que era min_samples_leaf=10. Que no final de cada folha tivesse
# pelo menos 10 amostras. Provalmente isso diminui as probabilidade de cair no overthing. 
# Essa tecnica, ela "poda" e evita que a arvore crie divisões para poucos registros. (Já não é recomendada para grande
# datasets)
# Isso indica que, apesar de mais controlada que a árvore sem restrições, 
# ela ainda apresenta maior tendência ao sobreajuste do que a Árvore Restrita.
# A Árvore Otimizada é melhor do que a Árvore Selvagem, 
# mas até agora a Árvore Restrita continua sendo a opção mais estável.
# =============================================================================================================== #

# arvores_otimizadas.py 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def treinar_arvore_otimizada(X_train, X_test, y_train, y_test):
    """
    Treina e avalia uma Árvore de Decisão Otimizada.
    """

    print("\n" + "=" * 70)
    print("========== ÁRVORE OTIMIZADA ==============")
    print("=" * 70)

    print("\nConfiguração utilizada:")
    # ================================================
    # Critério utilizado para medir a qualidade das divisões,
    # O critério "gini" mede a impureza dos nós da árvore.
    # A árvore tenta dividir os dados de forma que os grupos fiquem
    # ================================================
    print("- criterion: gini")

    # ================================================
    # Técnica de pré-poda usando quantidade mínima de amostras por folha,
    # Isso evita que a árvore crie folhas muito pequenas e específicas,
    # ================================================
    print("- min_samples_leaf: 10")

    # ================================================
    # Controle da aleatoriedade do modelo
    # O random_state=42 garante que o resultado seja reproduzível.
    # ================================================
    print("- random_state: 42")

    # ==============================================================================
    # Criação do modelo com pré-poda por quantidade mínima de amostras na folha
    # O parâmetro "min_samples_leaf=10" impede que a árvore crie folhas finais
    # com poucos registros. 
    # obs: Suspeito que ela não vai treinar direito
    # ==============================================================================
    modelo = DecisionTreeClassifier(
        min_samples_leaf=10,
        random_state=42
    )

    # ==============================================================================
    # Treinamento do modelo
    # O modelo aprende as regras de decisão utilizando os dados de treino.
    # ==============================================================================
    modelo.fit(X_train, y_train)

    # ==============================================================================
    # Geração das previsões
    # As previsões são feitas no treino e no teste para comparar o desempenho
    # em dados conhecidos e em dados novos.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo das métricas de avaliação
    # acuracia_treino = mede o desempenho nos dados usados para aprendizagem.
    # acuracia_teste  =  mede o desempenho em dados não vistos pelo modelo.
    # diferenca = acuracia_treino - acuracia_teste = ajuda a identificar sinais de overfitting.
    # ==============================================================================

    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    diferenca = acuracia_treino - acuracia_teste

    print("\nResultados:")
    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Diferença treino-teste: {diferenca * 100:.2f}%")

    return modelo