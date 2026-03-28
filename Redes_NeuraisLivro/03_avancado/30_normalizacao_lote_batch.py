import numpy as np
class BN:
    def __init__(self, d):
        self.g, self.b = np.ones(d), np.zeros(d)
    def forward(self, x):
        mu, var = x.mean(0), x.var(0)
        return self.g * (x - mu) / np.sqrt(var + 1e-8) + self.b
if __name__ == "__main__":
    bn = BN(10)
    print(bn.forward(np.random.randn(5, 10)).shape)
