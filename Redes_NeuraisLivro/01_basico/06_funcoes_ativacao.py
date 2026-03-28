import numpy as np
def sigmoid(x): return 1 / (1 + np.exp(-x))
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)
if __name__ == "__main__":
    x = np.array([-1, 0, 1])
    print(f"Sigmoid: {sigmoid(x)}")
    print(f"ReLU: {relu(x)}")
