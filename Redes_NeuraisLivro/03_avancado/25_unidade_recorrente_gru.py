import numpy as np
class GRU:
    def __init__(self, i, h):
        self.wr = np.random.randn(h, i + h)
        self.wz = np.random.randn(h, i + h)
        self.wh = np.random.randn(h, i + h)
    def forward(self, x, h):
        xh = np.vstack((h, x))
        r = 1 / (1 + np.exp(-np.dot(self.wr, xh)))
        z = 1 / (1 + np.exp(-np.dot(self.wz, xh)))
        xh_r = np.vstack((r * h, x))
        h_p = np.tanh(np.dot(self.wh, xh_r))
        h = (1 - z) * h + z * h_p
        return h
if __name__ == "__main__":
    gru = GRU(10, 20)
    h = np.zeros((20, 1))
    x = np.random.randn(10, 1)
    print(gru.forward(x, h).shape)
