import numpy as np
def sig(x): return 1/(1+np.exp(-x))
def dsig(x): return x*(1-x)
x, y = np.array([[0.5, 0.1]]), np.array([[0.7]])
w1 = np.random.randn(2, 3)
w2 = np.random.randn(3, 1)
for _ in range(1000):
    z1 = np.dot(x, w1)
    a1 = sig(z1)
    z2 = np.dot(a1, w2)
    a2 = sig(z2)
    erro = y - a2
    d_a2 = erro * dsig(a2)
    d_w2 = a1.T.dot(d_a2)
    d_a1 = d_a2.dot(w2.T) * dsig(a1)
    d_w1 = x.T.dot(d_a1)
    w1 += d_w1 * 0.1
    w2 += d_w2 * 0.1
print(a2)
