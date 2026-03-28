import numpy as np
def sig(x): return 1/(1+np.exp(-x))
def dsig(x): return x*(1-x)
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
wh = np.random.uniform(size=(2,2))
bh = np.random.uniform(size=(1,2))
wo = np.random.uniform(size=(2,1))
bo = np.random.uniform(size=(1,1))
for _ in range(10000):
    h = sig(np.dot(x, wh) + bh)
    o = sig(np.dot(h, wo) + bo)
    eo = y - o
    do = eo * dsig(o)
    eh = do.dot(wo.T)
    dh = eh * dsig(h)
    wo += h.T.dot(do) * 0.1
    bo += np.sum(do, axis=0, keepdims=True) * 0.1
    wh += x.T.dot(dh) * 0.1
    bh += np.sum(dh, axis=0, keepdims=True) * 0.1
print(o)
