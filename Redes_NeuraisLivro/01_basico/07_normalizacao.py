import numpy as np
def normalizar(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))
if __name__ == "__main__":
    x = np.array([10, 20, 30, 40, 50])
    print(normalizar(x))
