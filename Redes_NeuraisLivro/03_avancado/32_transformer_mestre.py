import numpy as np
def sm(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)
class MHA:
    def __init__(self, dm, nh):
        self.dm, self.nh, self.dk = dm, nh, dm // nh
        self.wq, self.wk, self.wv, self.wo = [np.random.randn(dm, dm) * 0.1 for _ in range(4)]
    def forward(self, q, k, v):
        bs = q.shape[0]
        q = np.dot(q, self.wq).reshape(bs, -1, self.nh, self.dk).transpose(0, 2, 1, 3)
        k = np.dot(k, self.wk).reshape(bs, -1, self.nh, self.dk).transpose(0, 2, 1, 3)
        v = np.dot(v, self.wv).reshape(bs, -1, self.nh, self.dk).transpose(0, 2, 1, 3)
        s = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.dk)
        a = sm(s)
        return np.matmul(a, v).transpose(0, 2, 1, 3).reshape(bs, -1, self.dm).dot(self.wo)
if __name__ == "__main__":
    mha = MHA(512, 8)
    x = np.random.randn(1, 10, 512)
    print(mha.forward(x, x, x).shape)
