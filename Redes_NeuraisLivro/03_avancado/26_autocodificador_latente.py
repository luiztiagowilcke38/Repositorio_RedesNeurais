import numpy as np
class AE:
    def __init__(self, i, h):
        self.we = np.random.randn(i, h)
        self.wd = np.random.randn(h, i)
    def forward(self, x):
        z = np.tanh(np.dot(x, self.we))
        r = np.tanh(np.dot(z, self.wd))
        return r
if __name__ == "__main__":
    ae = AE(10, 2)
    print(ae.forward(np.random.randn(1, 10)).shape)
