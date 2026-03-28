import numpy as np
class RNN:
    def __init__(self, i, h, o):
        self.wxh = np.random.randn(h, i) * 0.01
        self.whh = np.random.randn(h, h) * 0.01
        self.why = np.random.randn(o, h) * 0.01
        self.bh = np.zeros((h, 1))
        self.by = np.zeros((o, 1))
    def step(self, x, h):
        h = np.tanh(np.dot(self.wxh, x) + np.dot(self.whh, h) + self.bh)
        y = np.dot(self.why, h) + self.by
        return y, h
if __name__ == "__main__":
    rnn = RNN(10, 20, 10)
    h = np.zeros((20, 1))
    x = np.random.randn(10, 1)
    y, h = rnn.step(x, h)
    print(y.shape)
