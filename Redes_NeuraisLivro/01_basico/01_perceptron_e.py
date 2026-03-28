import numpy as np
def perceptron_e():
    entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
    saidas = np.array([0,0,0,1])
    pesos = np.array([0.0,0.0])
    vies = 0.0
    taxa = 0.1
    for _ in range(10):
        for i in range(4):
            pred = 1 if np.dot(entradas[i], pesos) + vies > 0 else 0
            erro = saidas[i] - pred
            pesos += taxa * erro * entradas[i]
            vies += taxa * erro
    for i in range(4):
        res = 1 if np.dot(entradas[i], pesos) + vies > 0 else 0
        print(f"{entradas[i]} -> {res}")
if __name__ == "__main__":
    perceptron_e()
