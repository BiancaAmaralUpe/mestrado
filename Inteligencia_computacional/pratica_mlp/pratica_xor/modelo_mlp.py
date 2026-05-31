# pyrefly: ignore [missing-import]
from sklearn.neural_network import MLPClassifier

# pyrefly: ignore [missing-import]
from .constants import (
    CAMADAS_OCULTAS_MLP,
    FUNCAO_ATIVACAO_MLP,
    SOLVER_MLP,
    ALPHA_MLP,
    QUANTIDADE_MAXIMA_ITERACOES_MLP,
    RANDOM_STATE_MLP,
)

# ======================================================================================== #
# Responsável por criar o modelo MLP usado para resolver o problema XOR.
#
# Diferente do Perceptron simples, a MLP possui camada oculta e função de ativação
# não linear. Isso permite que ela aprenda uma fronteira de decisão não linear,
# necessária para resolver o XOR.
# ======================================================================================== #
def criar_modelo_mlp() -> MLPClassifier:
    """
    Cria uma MLP para resolver o problema XOR.

    A MLP usada possui:
        - uma camada oculta;
        - ativação não linear tanh;
        - solver lbfgs;
        - saída binária para classificar as classes 0 e 1.
    """
    # ======================================================================================== #
    # Cria a MLP com uma camada oculta.
    #
    # A camada oculta permite que o modelo transforme os dados de entrada.
    # Essa transformação é o que torna possível resolver o XOR, já que o problema
    # não pode ser separado por uma única linha reta no espaço original.
    # ======================================================================================== #
    modelo_mlp = MLPClassifier(
        hidden_layer_sizes=CAMADAS_OCULTAS_MLP,
        activation=FUNCAO_ATIVACAO_MLP,
        solver=SOLVER_MLP,
        alpha=ALPHA_MLP,
        max_iter=QUANTIDADE_MAXIMA_ITERACOES_MLP,
        random_state=RANDOM_STATE_MLP,
    )

    return modelo_mlp