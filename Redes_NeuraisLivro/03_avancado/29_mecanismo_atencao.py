import numpy as np
def attn(q, k, v):
    s = np.dot(q, k.T) / np.sqrt(k.shape[-1])
    w = np.exp(s - np.max(s))
    w /= np.sum(w, axis=-1, keepdims=True)
    return np.dot(w, v)
if __name__ == "__main__":
    q = k = v = np.random.randn(1, 5)
    print(attn(q, k, v))
