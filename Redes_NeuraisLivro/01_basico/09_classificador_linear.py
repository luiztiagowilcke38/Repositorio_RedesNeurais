import numpy as np
class Classificador:
    def __init__(self, d=2): self.w = np.random.randn(d)
    def pred(self, x): return 1 if np.dot(x, self.w) > 0 else 0
if __name__ == "__main__":
    c = Classificador()
    print(c.pred(np.array([0.5, -0.2])))
