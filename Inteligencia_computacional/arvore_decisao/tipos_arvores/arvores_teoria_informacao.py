# ================================================================================ #
#                                   ARVORES                                        #
#                                   TEORIA DA INFORMAÇÃO                           #
# ================================================================================ #
# ============================================================================================================== #
# esse modelo, ele overfithing. Provavemente porque por causa disso : print("- max_depth: None")
# Ela também teve um bom equilibrio entre desempenho e generalização. teve uma acuracia semelhante ao da arvore_otimizada
# Mas com uma diferença treino-teste bem menor. O modelo memorizou os dados de treinamento 
# conseguindo manter um desempenho mais estavel em dados não vistos. Mas também também, vai continuar em 
# não recomendados para Producao pois mostrou um forte overfithing.
# 
# 
# =============================================================================================================== #

# arvores_teoria_informacao.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def treinar_arvore_teoria_informacao(X_train, X_test, y_train, y_test):
    """
    Treina e avalia uma Árvore de Decisão utilizando Teoria da Informação.

    Nesta configuração, o critério utilizado é a entropia.
    A entropia mede o grau de impureza ou desorganização dos dados em cada nó.
    """

    print("\n" + "=" * 70)
    print("========== ÁRVORE COM TEORIA DA INFORMAÇÃO ==============")
    print("=" * 70)

    print("\nConfiguração utilizada:")

    # ================================================
    # Critério baseado em teoria da informação
    # O critério "entropy" utiliza a entropia para medir a impureza dos nós.
    # A árvore escolhe as divisões que mais reduzem a desorganização das classes,
    # buscando separar melhor os pacientes com e sem indicação de diabetes.
    # ================================================
    print("- criterion: entropy")

    # ================================================
    # Profundidade sem limitador
    # O objetivo aqui é observar o comportamento do critério de entropia
    # ================================================
    print("- max_depth: None")

    # ================================================
    # Controle da aleatoriedade do modelo
    # ================================================
    # O random_state=42 garante que os resultados sejam reproduzíveis
    # sempre que o código for executado novamente.
    print("- random_state: 42")

    # ==============================================================================
    # Criação do modelo utilizando o critério de entropia
    # Nesta árvore, o modelo utiliza a entropia para decidir as melhores divisões.
    # A entropia é uma medida da teoria da informação que indica o grau de mistura
    # das classes em um nó.
    # ==============================================================================
    modelo = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=3,
        random_state=42
    )

    # ==============================================================================
    # Treinamento do modelo
    # ==============================================================================
    # O modelo aprende as regras de decisão a partir dos dados de treino.
    modelo.fit(X_train, y_train)

    # ==============================================================================
    # Geração das previsões
    # As previsões são feitas no conjunto de treino e no conjunto de teste.
    # Isso permite comparar o desempenho em dados conhecidos e em dados novos.
    # ==============================================================================
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # ==============================================================================
    # Cálculo das métricas de avaliação
    # acuracia_treino =  mostra o quanto o modelo acertou nos dados usados
    # durante o aprendizado.
    # acuracia_teste =  mostra o desempenho em dados que o modelo não viu.
    # diferenca = acuracia_treino - acuracia_teste =  ajuda a identificar possível overfitting.
    # ==============================================================================

    acuracia_treino = accuracy_score(y_train, y_pred_train)
    acuracia_teste = accuracy_score(y_test, y_pred_test)

    diferenca = acuracia_treino - acuracia_teste

    print("\nResultados:")
    print(f"Acurácia no treino: {acuracia_treino * 100:.2f}%")
    print(f"Acurácia no teste: {acuracia_teste * 100:.2f}%")
    print(f"Diferença treino-teste: {diferenca * 100:.2f}%")

    return modelo