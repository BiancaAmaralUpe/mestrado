# caso_xor.py 

#============================================================================================#
#  Quando há duas entradas diferentes, o Perceptron não consegue resolver o problema porque
#  ele não é linearmente separável. 
#  Perceptron atual consegue aprender AND e OR, porque eles são problemas linearmente separáveis.
#  Não dá para separar os pontos do XOR com uma única linha reta.
#============================================================================================#

# pyrefly: ignore [missing-import]
import numpy as np

from .perceptrons import Perceptron

def caso_xor():
    # Entradas da porta lógica XOR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Respostas corretas da porta lógica XOR
    # 0 XOR 0 = 0
    # 0 XOR 1 = 1
    # 1 XOR 0 = 1
    # 1 XOR 1 = 0
    y = np.array([0, 1, 1, 0])

    modelo = Perceptron(learning_rate=0.1, n_epochs=10)

    # Treina o modelo
    modelo.fit(X, y)

    # Faz as previsões
    previsoes = modelo.predict(X)

    print("Caso XOR")
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