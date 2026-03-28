import numpy as np
class LSTM:
    def __init__(self, i, h):
        self.wf = np.random.randn(h, i + h)
        self.wi = np.random.randn(h, i + h)
        self.wc = np.random.randn(h, i + h)
        self.wo = np.random.randn(h, i + h)
    def forward(self, x, h, c):
        xh = np.vstack((h, x))
        f = 1 / (1 + np.exp(-np.dot(self.wf, xh)))
        i = 1 / (1 + np.exp(-np.dot(self.wi, xh)))
        c_p = np.tanh(np.dot(self.wc, xh))
        c = f * c + i * c_p
        o = 1 / (1 + np.exp(-np.dot(self.wo, xh)))
        h = o * np.tanh(c)
        return h, c
if __name__ == "__main__":
    lstm = LSTM(10, 20)
    h, c = np.zeros((20,1)), np.zeros((20,1))
    x = np.random.randn(10,1)
    h, c = lstm.forward(x, h, c)
    print(h.shape)
