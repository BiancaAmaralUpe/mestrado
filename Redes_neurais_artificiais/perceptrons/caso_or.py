# caso_or.py

# pyrefly: ignore [missing-import]
import numpy as np

from .perceptrons import Perceptron

def caso_or():
    # Entradas da porta lógica OR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Respostas corretas da porta lógica OR
    # 0 OR 0 = 0
    # 0 OR 1 = 1
    # 1 OR 0 = 1
    # 1 OR 1 = 1
    y = np.array([0, 1, 1, 1])

    modelo = Perceptron(learning_rate=0.1, n_epochs=10)

    # Treina o modelo
    modelo.fit(X, y)

    # Faz as previsões
    previsoes = modelo.predict(X)

    print("Caso OR")
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