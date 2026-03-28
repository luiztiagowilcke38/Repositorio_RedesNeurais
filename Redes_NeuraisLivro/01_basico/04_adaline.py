import numpy as np
class Adaline:
    def __init__(self, n=2, t=0.01):
        self.w = np.zeros(n + 1)
        self.t = t
    def treinar(self, x, y, e=50):
        for _ in range(e):
            for xi, target in zip(x, y):
                out = self.net(xi)
                erro = target - out
                self.w[1:] += self.t * erro * xi
                self.w[0] += self.t * erro
    def net(self, x): return np.dot(x, self.w[1:]) + self.w[0]
    def pred(self, x): return 1 if self.net(x) >= 0 else -1
if __name__ == "__main__":
    x = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
    y = np.array([1,-1,-1,-1])
    ada = Adaline()
    ada.treinar(x, y)
    print(ada.w)
