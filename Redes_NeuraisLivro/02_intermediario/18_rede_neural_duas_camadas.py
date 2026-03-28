import numpy as np
class Rede:
    def __init__(self, i, h, o):
        self.w1 = np.random.randn(i, h)
        self.w2 = np.random.randn(h, o)
    def forward(self, x):
        self.h = np.tanh(np.dot(x, self.w1))
        return np.dot(self.h, self.w2)
if __name__ == "__main__":
    r = Rede(2, 4, 1)
    print(r.forward(np.array([[1, 0]])))
