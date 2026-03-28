import numpy as np
class TL:
    def __init__(self, pre):
        self.base = pre
        self.head = np.random.randn(pre.shape[1], 1)
    def forward(self, x):
        feat = np.dot(x, self.base)
        return np.dot(feat, self.head)
if __name__ == "__main__":
    b = np.random.randn(10, 5)
    tl = TL(b)
    print(tl.forward(np.random.randn(1, 10)).shape)
