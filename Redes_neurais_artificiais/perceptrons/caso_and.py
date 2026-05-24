# pyrefly: ignore [missing-import]
import numpy as np

from .perceptrons import Perceptron

#============================================================================================#
# Treina e testa um Perceptron para representar a porta lógica AND.
#============================================================================================#

def caso_and():
    # ==============================================================================#
    # Entradas da porta lógica AND
    # ==============================================================================#
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    # ================================================================================#
    # Respostas corretas da porta lógica AND
    # ================================================================================# 
    y = np.array([0, 0, 0, 1])

    modelo = Perceptron(learning_rate=0.1, n_epochs=10)

    # ================================================================================# 
    # Treina o modelo
    # ================================================================================# 
    modelo.fit(X, y)

    # ================================================================================# 
    # Faz as previsões
    # ================================================================================# 
    previsoes = modelo.predict(X)

    print("Caso AND")
    print("====================================")

    print("Entradas:")
    print(X)

    print("\nSaídas corretas:")
    print(y)

    print("\nPrevisões do modelo:")
    print(previsoes)

    print("\nPesos aprendidos:")
    print(modelo.pesos)

    print("\nViés aprendido:")
    print(modelo.vies)

    return modelo, previsoes


if __name__ == "__main__":
    caso_and()